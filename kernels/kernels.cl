
__kernel void matrix_dot_vector(__global const float4 *matrix,
__global const float4 *vector, __global float *result)
{
    int gid = get_global_id(0);
    result[gid] = dot(matrix[gid], vector[0]);
}

__kernel void square_sum(__global float *template, __global float *frame, __global float *output)
{
    int gid = get_global_id(0);
    output[gid] = pow((template[gid] - frame[gid]),2);
}