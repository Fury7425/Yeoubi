
// --- Yeoubi: GGML_OP_ACT_QUANT (appended by runtime/cmake/patch_ggml.cmake) ---
// Activation fake-quant per row, one threadgroup per row. Reuses the argmax
// kargs: ne00 = row length, nb01 = row stride of all three tensors,
// npart = mode | has_mask << 4. Modes and arithmetic as act_quant.comp.
kernel void kernel_act_quant_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device   const char * src1,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int  mode   = (int) (args.npart & 15);
    const bool masked = ((args.npart >> 4) & 1) != 0;
    device const float * x = (device const float *) (src0 + tgpig*args.nb01);
    device const float * m = (device const float *) (src1 + tgpig*args.nb01);
    device       float * y = (device       float *) (dst  + tgpig*args.nb01);

    float acc = 0.0f;
    for (int i = tpitg; i < args.ne00; i += ntg) {
        const float v = (masked && !(m[i] > -INFINITY)) ? 0.0f : x[i];
        acc = mode == 0 ? acc + fabs(v) : max(acc, fabs(v));
    }
    acc = mode == 0 ? simd_sum(acc) : simd_max(acc);

    threadgroup float * sh = (threadgroup float *) shmem;
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            sh[tiisg] = 0.0f;  // |x| >= 0, so 0 is neutral for sum and max
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            sh[sgitg] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = sh[tiisg];
        acc = mode == 0 ? simd_sum(acc) : simd_max(acc);
    }

    const float stat  = max(mode == 0 ? precise::divide(acc, (float) args.ne00) : acc, 1e-8f);
    const float level = mode == 2 ? 127.0f : 7.0f;
    const float s     = precise::divide(1.0f, stat) * (mode == 0 ? 2.6457513110645907f : level);

    for (int i = tpitg; i < args.ne00; i += ntg) {
        const float v = (masked && !(m[i] > -INFINITY)) ? 0.0f : x[i];
        y[i] = precise::divide(clamp(rint(v*s), -level, level), s);
    }
}
