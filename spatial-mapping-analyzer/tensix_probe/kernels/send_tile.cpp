#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_x = get_arg_val<uint32_t>(0);
    const uint32_t dst_y = get_arg_val<uint32_t>(1);
    const uint32_t semaphore = get_semaphore(get_arg_val<uint32_t>(2));

    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t remote_input_cb = get_compile_time_arg_val(1);

    cb_wait_front(output_cb, 1);
    auto local_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore);
    noc_semaphore_wait(local_sem, 1);

    const uint32_t bytes = get_tile_size(output_cb);
    const uint32_t src_addr = get_read_ptr(output_cb);
    const uint32_t remote_cb_addr = get_read_ptr(remote_input_cb);
    const uint64_t dst_addr = get_noc_addr(dst_x, dst_y, remote_cb_addr);

    noc_async_write(src_addr, dst_addr, bytes);
    noc_async_write_barrier();

    const uint64_t remote_sem = get_noc_addr(dst_x, dst_y, semaphore);
    noc_semaphore_inc(remote_sem, 1);
    noc_async_atomic_barrier();
    noc_semaphore_set(local_sem, 0);

    cb_pop_front(output_cb, 1);
}
