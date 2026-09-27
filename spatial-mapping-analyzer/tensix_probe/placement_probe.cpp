// Minimal TT-Metal Tensix placement probe.
// Runs one BF16 tile add on the requested logical worker core and writes JSON.

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
    unsigned long parsed = std::stoul(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument(std::string("Invalid ") + name);
    }
    return static_cast<uint32_t>(parsed);
}

struct Args {
    uint32_t core_x = 0;
    uint32_t core_y = 0;
    std::filesystem::path result;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + key);
        }
        const std::string value = argv[++i];
        if (key == "--core-x") {
            args.core_x = parse_uint(value, "core-x");
        } else if (key == "--core-y") {
            args.core_y = parse_uint(value, "core-y");
        } else if (key == "--result") {
            args.result = value;
        } else {
            throw std::invalid_argument("Unknown argument: " + key);
        }
    }
    if (args.result.empty()) {
        throw std::invalid_argument("--result is required");
    }
    return args;
}

std::filesystem::path kernel_path(const char* relative) {
    const char* home = std::getenv("TT_METAL_HOME");
    if (home == nullptr || std::string(home).empty()) {
        throw std::runtime_error("TT_METAL_HOME is required");
    }
    return std::filesystem::path(home) / relative;
}

void write_result(const std::filesystem::path& path, bool passed, uint32_t x, uint32_t y) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open result path");
    }
    output << "{\n"
           << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
           << "  \"core\": [" << x << ", " << y << "],\n"
           << "  \"elements\": " << (tt::constants::TILE_WIDTH * tt::constants::TILE_WIDTH) << "\n"
           << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const CoreCoord core{args.core_x, args.core_y};

        auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        Program program = CreateProgram();

        constexpr uint32_t n_elements = tt::constants::TILE_WIDTH * tt::constants::TILE_WIDTH;
        constexpr uint32_t tile_size = sizeof(bfloat16) * n_elements;
        constexpr uint32_t num_tiles = 1;

        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = tile_size,
            .buffer_type = BufferType::DRAM,
        };
        distributed::ReplicatedBufferConfig replicated_config{.size = tile_size};

        auto src0 = distributed::MeshBuffer::create(replicated_config, dram_config, mesh_device.get());
        auto src1 = distributed::MeshBuffer::create(replicated_config, dram_config, mesh_device.get());
        auto dst = distributed::MeshBuffer::create(replicated_config, dram_config, mesh_device.get());

        auto cb_config = [&](CBIndex index) {
            return CircularBufferConfig(num_tiles * tile_size, {{index, DataFormat::Float16_b}})
                .set_page_size(index, tile_size);
        };
        CreateCircularBuffer(program, core, cb_config(CBIndex::c_0));
        CreateCircularBuffer(program, core, cb_config(CBIndex::c_1));
        CreateCircularBuffer(program, core, cb_config(CBIndex::c_16));

        auto reader = CreateKernel(
            program,
            kernel_path("tt_metal/programming_examples/add_2_integers_in_compute/kernels/dataflow/reader_binary_1_tile.cpp").string(),
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto writer = CreateKernel(
            program,
            kernel_path("tt_metal/programming_examples/add_2_integers_in_compute/kernels/dataflow/writer_1_tile.cpp").string(),
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        auto compute = CreateKernel(
            program,
            kernel_path("tt_metal/programming_examples/add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp").string(),
            core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});

        std::vector<bfloat16> a(n_elements, bfloat16(1.0f));
        std::vector<bfloat16> b(n_elements, bfloat16(2.0f));
        EnqueueWriteMeshBuffer(cq, src0, a, false);
        EnqueueWriteMeshBuffer(cq, src1, b, false);

        SetRuntimeArgs(program, reader, core, {
            static_cast<uint32_t>(src0->address()),
            static_cast<uint32_t>(src1->address()),
        });
        SetRuntimeArgs(program, compute, core, {});
        SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(dst->address())});

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        std::vector<bfloat16> result;
        distributed::EnqueueReadMeshBuffer(cq, result, dst, true);

        bool passed = result.size() == n_elements;
        if (passed) {
            for (const auto& value : result) {
                if (std::abs(static_cast<float>(value) - 3.0f) > 0.3f) {
                    passed = false;
                    break;
                }
            }
        }

        write_result(args.result, passed, args.core_x, args.core_y);
        mesh_device->close();
        return passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "spatial_tensix_probe: " << error.what() << "\n";
        return 1;
    }
}
