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

__kernel void connected_components(const int w, const int h, __global int *labels)
{
    int x = get_global_id(1);
    int y = get_global_id(0);
    if (x == 0 || y == 0 || x >= w-1 || y >= h-1) return;
    int idx = y*(w*2) + x*2;
    if(labels[idx] == 0) return;

    int min_label = get_min_label(labels, idx, w, h);
    if (min_label != 0)
    {
        labels[idx] = min_label;
    }
}

int get_min_label(global int *labels, int idx, int w, int h) { // returns the smallest label stored in the connected adjacent pixels.
  int min_label = labels[idx];

  for(int y=-1;y<=1;y++) {
    for(int x=-1;x<=1;x++) {
      int idx_m = idx + y*(w*2) + x*2;
      if (labels[idx_m] != 0 && labels[idx_m] < min_label) min_label = labels[idx_m];
    }
  }
  return min_label;
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

__kernel void hsv_mask(read_only image2d_t src, __global const float4 *mask, write_only image2d_t dest){
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    uint4 pix = read_imageui(src, sampler, pos);

    if((pix.x < mask[0].s0) || (pix.x > mask[1].s0) || (pix.y < mask[0].s1)){
        pix.x = 0;
        pix.y = 0;
        pix.z = 0;
        //pix.a = 0;
    }

    write_imageui(dest, pos, pix);
}
__kernel void hsv_bin_mask(read_only image2d_t src, __global const float4 *mask, write_only image2d_t dest){
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    uint4 pix = read_imageui(src, sampler, pos);

    if((pix.x < mask[0].s0) || (pix.x > mask[1].s0) ||
        (pix.y < mask[0].s1) || (pix.y > mask[1].s1) ||
        (pix.z < mask[0].s2) || (pix.z > mask[1].s2)){
        pix.x = 255;
        pix.y = 255;
        pix.z = 255;
        // pix.a = 0;
    }
    else{
        pix.x = 0;
        pix.y = 0;
        pix.z = 0;
    }

    write_imageui(dest, pos, pix);
}

__kernel void merge_bin(read_only image2d_t src, read_only image2d_t src2, write_only image2d_t dest){
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    uint4 pix1 = read_imageui(src, sampler, pos);
    uint4 pix2 = read_imageui(src2, sampler, pos);

    if (pix1.x == 0 || pix2.x == 0){
        pix1.x = 0;
        pix1.y = 0;
        pix1.z = 0;
    }
    else{
        pix1.x = 255;
        pix1.y = 255;
        pix1.z = 255;
    }

    write_imageui(dest, pos, pix1);
}