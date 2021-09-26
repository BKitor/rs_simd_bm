use super::*;
#[test]
fn test_correctness() {
    let dim = 64;
    let a = build_vec(dim);
    let b = build_vec(dim);
    let kt_simd_a = KernelType::F32x4Vec(vec_to_f32x4(&a));
    let kt_simd_b = KernelType::F32x4Vec(vec_to_f32x4(&b));
    let kt_tst = sum_f32x4(&kt_simd_a, &kt_simd_b, dim);
    let kt_a = KernelType::F32Vec(a);
    let kt_b = KernelType::F32Vec(b);
    let kt_gt = sum_serial(&kt_a, &kt_b, dim);

    if let (
        KernelType::F32x4Vec(tst),
        KernelType::F32Vec(gt),
        KernelType::F32x4Vec(simd_a),
        KernelType::F32Vec(a),
    ) = (kt_tst, kt_gt, kt_simd_a, kt_a)
    {
        let v_tst = f32x4_to_vec(&tst);

        assert_eq![a, f32x4_to_vec(&simd_a)];
        assert_eq![gt, v_tst];
    } else {
        panic!("Converting one of the many things out of KT failed")
    }
}
