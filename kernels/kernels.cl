#define E .0000001f

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

__kernel void connected_components(__global int *labels,
__global int *labels_cc)
{
    int w = 1920;
    int h = 1080;
    int x = get_global_id(1);
    int y = get_global_id(0);
    int padding = w;
    int idx = y*(w*2) + x*2;
    //labels_cc[idx] = idx;
    int mask[9];

    //3x3 matrix with center of labels[idx]
    mask[0] = labels[idx - 2*w - 2];
    mask[1] = labels[idx - 2*w];
    mask[2] = labels[idx - 2*w + 2];
    mask[3] = labels[idx - 2];
    mask[4] = labels[idx];
    mask[5] = labels[idx + 2];
    mask[6] = labels[idx + 2*w - 2];
    mask[7] = labels[idx + 2*w];
    mask[8] = labels[idx + 2*w + 2];

    int min_label = labels[idx];
    for(int i=0; i<9; i++) {
        if(mask[i] < min_label && mask[i] > 0){
            min_label = mask[i];
        }
    }

    labels[idx] = min_label;
    labels_cc[idx] = labels[idx];

}

__kernel void rgb2hsv(read_only image2d_t src, write_only image2d_t dest){
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    uint4 pix = read_imageui(src, sampler, pos);
    // A simple test operation: delete pixel in form of a checkerboard pattern
    float x = (float)pix.x/255;
    float y = (float)pix.y/255;
    float z = (float)pix.z/255;

    float cmax = fmax(x,y);
    cmax = fmax(cmax, z);
    float cmin = fmin(x,y);
    cmin = fmin(cmin, z);
    float diff = cmax-cmin;

    float v = cmax;
    float h = 0;
    float s = 0;

    if (cmin == cmax)
        h = 0;
    else if (cmax == x)
        h = (unsigned int)( 0 + 43 * (y - z) / diff);
    else if (cmax == y)
        h = (unsigned int)(85 + 43 * (z - x) / diff);
    else if (cmax == z)
        h = (unsigned int)(171 + 43 * (x - y) / diff);

    if (cmax == 0)
        s = 0;
    else
        s = (float)(diff/cmax);

    pix.x = (int)(h);
    pix.y = (int)(s*255);
    pix.z = (int)(v*255);
    write_imageui(dest, pos, pix);
}

__kernel void hsvMask(read_only image2d_t src, __global const float4 *mask, write_only image2d_t dest){
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    uint4 pix = read_imageui(src, sampler, pos);

    if((pix.x < mask[0][0]) || (pix.x > mask[1][0]) || (pix.y < mask[0][1])){
        pix.x = 0;
        pix.y = 0;
        pix.z = 0;
        pix.a = 0;
    }
    write_imageui(dest, pos, pix);
}
