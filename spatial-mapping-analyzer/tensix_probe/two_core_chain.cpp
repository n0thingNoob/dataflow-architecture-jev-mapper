// Two-stage Tensix producer-consumer chain.
// Computes (A + B) on producer core, sends the intermediate directly over NoC,
// then computes intermediate + C on consumer core. The intermediate never returns to host.

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

uint32_t parse_uint(const std::string& value, const char* name) {
    size_t consumed = 0;
    const unsigned long parsed = std::stoul(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument(std::string("Invalid ") + name);
    }
    return static_cast<uint32_t>(parsed);
}

struct Args {
    CoreCoord producer{0, 0};
    CoreCoord consumer{1, 0};
    std::filesystem::path result;
    std::filesystem::path kernel_root;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + key);
        }
        const std::string value = argv[++i];
        if (key == "--producer-x") {
            args.producer.x = parse_uint(value, "producer-x");
        } else if (key == "--producer-y") {
            args.producer.y = parse_uint(value, "producer-y");
        } else if (key == "--consumer-x") {
            args.consumer.x = parse_uint(value, "consumer-x");
        } else if (key == "--consumer-y") {
            args.consumer.y = parse_uint(value, "consumer-y");
        } else if (key == "--result") {
            args.result = value;
        } else if (key == "--kernel-root") {
            args.kernel_root = value;
        } else {
            throw std::invalid_argument("Unknown argument: " + key);
        }
    }
    if (args.result.empty() || args.kernel_root.empty()) {
        throw std::invalid_argument("--result and --kernel-root are required");
    }
    if (args.producer == args.consumer) {
        throw std::invalid_argument("Producer and consumer must use different cores");
    }
    return args;
}

std::filesystem::path tt_kernel(const char* relative) {
    const char* home = std::getenv("TT_METAL_HOME");
    if (home == nullptr || std::string(home).empty()) {
        throw std::runtime_error("TT_METAL_HOME is required");
    }
    return std::filesystem::path(home) / relative;
}

void write_result(const Args& args, bool passed) {
    std::ofstream output(args.result);
    if (!output) {
        throw std::runtime_error("Cannot open result path");
    }
    output << "{\n"
           << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
           << "  \"producer_core\": [" << args.producer.x << ", " << args.producer.y << "],\n"
           << "  \"consumer_core\": [" << args.consumer.x << ", " << args.consumer.y << "],\n"
           << "  \"intermediate_transport\": \"noc_direct\",\n"
           << "  \"intermediate_returned_to_host\": false,\n"
           << "  \"elements\": " << tt::constants::TILE_HW << "\n"
           << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        Program program = CreateProgram();

        const auto producer_physical = mesh_device->worker_core_from_logical_core(args.producer);
        const auto consumer_physical = mesh_device->worker_core_from_logical_core(args.consumer);
        const CoreRangeSet chain_cores({
            CoreRange(args.producer, args.producer),
            CoreRange(args.consumer, args.consumer),
        });

        constexpr uint32_t tile_elements = tt::constants::TILE_HW;
        constexpr uint32_t tile_size = sizeof(bfloat16) * tile_elements;

        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM,
        };
        distributed::ReplicatedBufferConfig replicated_config{.size = tile_size};

        auto a_buffer = distributed::MeshBuffer::create(replicated_config, dram_config, mesh_device.get());
        auto b_buffer = distributed::MeshBuffer::create(replicated_config, dram_config, mesh_device.get());
        auto c_buffer = distributed::MeshBuffer::create(replicated_config, dram_config, mesh_device.get());
        auto out_buffer = distributed::MeshBuffer::create(replicated_config, dram_config, mesh_device.get());

        auto cb_config = [&](CBIndex index) {
            return CircularBufferConfig(tile_size, {{index, DataFormat::Float16_b}})
                .set_page_size(index, tile_size);
        };
        CreateCircularBuffer(program, chain_cores, cb_config(CBIndex::c_0));
        CreateCircularBuffer(program, chain_cores, cb_config(CBIndex::c_1));
        CreateCircularBuffer(program, chain_cores, cb_config(CBIndex::c_16));

        const uint32_t semaphore = CreateSemaphore(program, chain_cores, 0);

        const auto stage0_reader = CreateKernel(
            program,
            tt_kernel("tt_metal/programming_examples/add_2_integers_in_compute/kernels/dataflow/reader_binary_1_tile.cpp").string(),
            args.producer,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        const auto stage0_compute = CreateKernel(
            program,
            tt_kernel("tt_metal/programming_examples/add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp").string(),
            args.producer,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});
        const auto stage0_sender = CreateKernel(
            program,
            (args.kernel_root / "send_tile.cpp").string(),
            args.producer,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {static_cast<uint32_t>(CBIndex::c_16), static_cast<uint32_t>(CBIndex::c_0)},
            });

        const auto stage1_receiver = CreateKernel(
            program,
            (args.kernel_root / "receive_and_read_addend.cpp").string(),
            args.consumer,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = {static_cast<uint32_t>(CBIndex::c_0), static_cast<uint32_t>(CBIndex::c_1)},
            });
        const auto stage1_compute = CreateKernel(
            program,
            tt_kernel("tt_metal/programming_examples/add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp").string(),
            args.consumer,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});
        const auto stage1_writer = CreateKernel(
            program,
            tt_kernel("tt_metal/programming_examples/add_2_integers_in_compute/kernels/dataflow/writer_1_tile.cpp").string(),
            args.consumer,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        std::vector<bfloat16> a(tile_elements, bfloat16(1.0f));
        std::vector<bfloat16> b(tile_elements, bfloat16(2.0f));
        std::vector<bfloat16> c(tile_elements, bfloat16(4.0f));
        distributed::EnqueueWriteMeshBuffer(cq, a_buffer, a, false);
        distributed::EnqueueWriteMeshBuffer(cq, b_buffer, b, false);
        distributed::EnqueueWriteMeshBuffer(cq, c_buffer, c, false);

        SetRuntimeArgs(
            program,
            stage0_reader,
            args.producer,
            {static_cast<uint32_t>(a_buffer->address()), static_cast<uint32_t>(b_buffer->address())});
        SetRuntimeArgs(program, stage0_compute, args.producer, {});
        SetRuntimeArgs(
            program,
            stage0_sender,
            args.producer,
            {consumer_physical.x, consumer_physical.y, semaphore});

        SetRuntimeArgs(
            program,
            stage1_receiver,
            args.consumer,
            {
                producer_physical.x,
                producer_physical.y,
                semaphore,
                static_cast<uint32_t>(c_buffer->address()),
            });
        SetRuntimeArgs(program, stage1_compute, args.consumer, {});
        SetRuntimeArgs(
            program,
            stage1_writer,
            args.consumer,
            {static_cast<uint32_t>(out_buffer->address())});

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        std::vector<bfloat16> result;
        distributed::EnqueueReadMeshBuffer(cq, result, out_buffer, true);
        bool passed = result.size() == tile_elements;
        if (passed) {
            for (const auto& value : result) {
                if (std::abs(static_cast<float>(value) - 7.0f) > 0.3f) {
                    passed = false;
                    break;
                }
            }
        }

        write_result(args, passed);
        mesh_device->close();
        return passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "spatial_tensix_chain_probe: " << error.what() << "\n";
        return 1;
    }
}
