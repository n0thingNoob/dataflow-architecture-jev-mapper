#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_x = get_arg_val<uint32_t>(0);
    const uint32_t src_y = get_arg_val<uint32_t>(1);
    const uint32_t semaphore = get_semaphore(get_arg_val<uint32_t>(2));
    const uint32_t addend_addr = get_arg_val<uint32_t>(3);

    constexpr uint32_t received_cb = get_compile_time_arg_val(0);
    constexpr uint32_t addend_cb = get_compile_time_arg_val(1);

    const uint32_t tile_size = get_tile_size(received_cb);
    cb_reserve_back(received_cb, 1);

    auto local_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore);
    const uint64_t source_sem = get_noc_addr(src_x, src_y, semaphore);
    noc_semaphore_inc(source_sem, 1);
    noc_async_atomic_barrier();

    noc_semaphore_wait(local_sem, 1);
    noc_semaphore_set(local_sem, 0);
    cb_push_back(received_cb, 1);

    const InterleavedAddrGenFast<true> addend = {
        .bank_base_address = addend_addr,
        .page_size = tile_size,
        .data_format = DataFormat::Float16_b,
    };
    cb_reserve_back(addend_cb, 1);
    const uint32_t addend_l1 = get_write_ptr(addend_cb);
    noc_async_read_page(0, addend, addend_l1);
    noc_async_read_barrier();
    cb_push_back(addend_cb, 1);
}
