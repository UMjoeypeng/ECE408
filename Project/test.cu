#include <cstdio>

#include "cpu-new-forward.h"
#include "gpu-new-forward.h"
#define BIG 256
constexpr int Batch = 64;
constexpr int Map_out = 8;
constexpr int Channel = 3;
constexpr int Height = BIG;
constexpr int Width = BIG;
constexpr int K = 3;
float input[Batch][Channel][Height][Width];
float mask[Map_out][Channel][K][K];
float output[Batch][Map_out][Height][Width];
float output2[Batch][Map_out][Height][Width];
    #define FOR(i, __) for (int i = 0; (i) < (__); (i)++)
int main() {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    FOR(b, Batch)
    FOR(c, Channel) FOR(h, Height) FOR(w, Width)
        input[b][c][h][w] = b + c + h + w;
    FOR(m, Map_out)
    FOR(c, Channel) FOR(p, K) FOR(q, K)
        mask[m][c][p][q] = m + c + p + q;
    auto G=GPUInterface();
    float *devin, *devout, *devmask;
    G.conv_forward_gpu_prolog((float *)output, (float *)input, (float *)mask, &devout, &devin, &devmask, Batch, Map_out, Channel, Height, Width, K);
    G.conv_forward_gpu(devout, devin, devmask, Batch, Map_out, Channel, Height, Width, K);
    G.conv_forward_gpu_epilog((float *)output, devout, devin, devmask, Batch, Map_out, Channel, Height, Width, K);
    conv_forward_cpu((float *)output2, (float *)input, (float *)mask, Batch, Map_out, Channel, Height, Width, K);
    FOR(b, Batch)
    FOR(m, Map_out) FOR(h, Height_out) FOR(w, Width_out) if (output[b][m][h][w] != output2[b][m][h][w])
        printf("%lf %lf\n", output[b][m][h][w], output2[b][m][h][w]);
}
