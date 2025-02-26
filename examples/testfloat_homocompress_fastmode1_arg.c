/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Jiajun Huang <jiajunhuang19990916@gmail.com>
 *  @date Oct, 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "ZCCL.h"
#ifdef _OPENMP
#include "omp.h"
#endif
#define ITERATIONS 1

struct timeval startTime;
struct timeval endTime; /* Start and end times */
struct timeval costStart; /*only used for recording the cost*/
double totalCost = 0;

void cost_start()
{
    totalCost = 0;
    gettimeofday(&costStart, NULL);
}

void cost_end()
{
    double elapsed;
    struct timeval costEnd;
    gettimeofday(&costEnd, NULL);
    elapsed = ((costEnd.tv_sec * 1000000 + costEnd.tv_usec)
                  - (costStart.tv_sec * 1000000 + costStart.tv_usec))
        / 1000000.0;
    totalCost += elapsed;
}

void add_vectors(float *result, const float *a, const float *b, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

void error_evaluation(float *oriData, float *decData, size_t nbEle, size_t cmpSize)
{
    size_t i = 0;
    float Max = 0, Min = 0, diffMax = 0;
    Max = oriData[0];
    Min = oriData[0];
    diffMax = fabs(decData[0] - oriData[0]);
    double sum1 = 0, sum2 = 0;
    for (i = 0; i < nbEle; i++) {
        sum1 += oriData[i];
        sum2 += decData[i];
    }
    double mean1 = sum1 / nbEle;
    double mean2 = sum2 / nbEle;

    double sum3 = 0, sum4 = 0;
    double sum = 0, prodSum = 0, relerr = 0;

    double maxpw_relerr = 0;
    for (i = 0; i < nbEle; i++) {
        if (Max < oriData[i])
            Max = oriData[i];
        if (Min > oriData[i])
            Min = oriData[i];

        float err = fabs(decData[i] - oriData[i]);
        if (oriData[i] != 0) {
            if (fabs(oriData[i]) > 1)
                relerr = err / oriData[i];
            else
                relerr = err;
            if (maxpw_relerr < relerr)
                maxpw_relerr = relerr;
        }

        if (diffMax < err)
            diffMax = err;
        prodSum += (oriData[i] - mean1) * (decData[i] - mean2);
        sum3 += (oriData[i] - mean1) * (oriData[i] - mean1);
        sum4 += (decData[i] - mean2) * (decData[i] - mean2);
        sum += err * err;
    }
    double std1 = sqrt(sum3 / nbEle);
    double std2 = sqrt(sum4 / nbEle);
    double ee = prodSum / nbEle;
    double acEff = ee / std1 / std2;

    double mse = sum / nbEle;
    double range = Max - Min;
    double psnr = 20 * log10(range) - 10 * log10(mse);
    double nrmse = sqrt(mse) / range;

    double compressionRatio = 1.0 * nbEle * sizeof(float) / cmpSize;

    printf("Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
    printf("Max absolute error = %.10f\n", diffMax);
    printf("Max relative error = %f\n", diffMax / (Max - Min));
    printf("Max pw relative error = %f\n", maxpw_relerr);
    printf("PSNR = %.3f, NRMSE= %.5G\n", psnr, nrmse);
    printf("acEff=%f\n", acEff);
    printf("Compression Ratio = %f\n", compressionRatio);
}

float computeValueRange_float(float *oriData, size_t length, float *radius, float *medianValue)
{
    float min = oriData[0];
    float max = oriData[0];
    for (size_t i = 0; i < length; i++) {
        float v = oriData[i];
        if (min > v)
            min = v;
        else if (max < v)
            max = v;
    }
    float valueRange = max - min;
    if (radius != NULL) {
        *radius = valueRange / 2;
        *medianValue = min + *radius;
    }

    return valueRange;
}

int main(int argc, char *argv[])
{
    char oriFilePath[640];
    char oriFilePath2[640];
    int status = 0;
    if (argc != 5) {
        printf(
            "Usage: testfloat_homocompress_fastmode1_arg [srcFilePath] [srcFilePath2] [blockSize] [relErrBound]\n");
        exit(0);
    }
    sprintf(oriFilePath, "%s", argv[1]);
    sprintf(oriFilePath2, "%s", argv[2]);
    int blockSize = atoi(argv[3]);
    float relerrorBound = atof(argv[4]);

    float *oriData = NULL;
    float *oriData2 = NULL;

    unsigned char *cmpData = NULL;
    unsigned char *cmpData2 = NULL;
    unsigned char *addcmpData = NULL;

    float *decData = NULL;
    float *decData2 = NULL;
    float *adddecData = NULL;
    size_t nbEle = 0;
    size_t cmpSize = 0;
    size_t cmpSize2 = 0;
    size_t addcmpSize = 0;
    size_t addcmpSize_normal = 0;
    oriData = readFloatData(oriFilePath, &nbEle, &status);
    oriData2 = readFloatData(oriFilePath2, &nbEle, &status);

    float *sub_oriData = (float *) calloc(nbEle, sizeof(float));
    cmpData = (unsigned char *) calloc(4 * nbEle, sizeof(unsigned char));

    cmpData2 = (unsigned char *) calloc(4 * nbEle, sizeof(unsigned char));
    addcmpData = (unsigned char *) calloc(4 * nbEle, sizeof(unsigned char));
    unsigned char *addcmpData_normal = (unsigned char *) calloc(4 * nbEle, sizeof(unsigned char));

    decData = (float *) calloc(nbEle, sizeof(float));
    decData2 = (float *) calloc(nbEle, sizeof(float));
    adddecData = (float *) calloc(nbEle, sizeof(float));
    float *adddecData_normal = (float *) calloc(nbEle, sizeof(float));
    float *adddecData_2 = (float *) calloc(nbEle, sizeof(float));
    float *addoriData = (float *) calloc(nbEle, sizeof(float));

    float valueRange = computeValueRange_float((float *) oriData, nbEle, NULL, NULL);
    float valueRange2 = computeValueRange_float((float *) oriData2, nbEle, NULL, NULL);
    float larger_range;

    if (valueRange > valueRange2) {
        larger_range = valueRange;
    } else {
        larger_range = valueRange2;
    }

    float errBound = larger_range * relerrorBound;

    for (size_t i = 0; i < ITERATIONS; i++) {
        add_vectors(addoriData, oriData, oriData2, nbEle);
    }

    double initial_com_cost = 0.0, normal_decom_cost = 0.0, normal_cpt_cost = 0.0,
           normal_com_cost = 0.0, normal_total_cost = 0.0, homo_total_cost = 0.0;

    cost_start();

    for (size_t i = 0; i < ITERATIONS; i++) {
        ZCCL_float_openmp_threadblock_arg(cmpData, oriData, &cmpSize, errBound, nbEle, blockSize);
        ZCCL_float_openmp_threadblock_arg(
            cmpData2, oriData2, &cmpSize2, errBound, nbEle, blockSize);
    }
    cost_end();
    initial_com_cost = totalCost / ITERATIONS;

    cost_start();

    for (size_t i = 0; i < ITERATIONS; i++) {
        ZCCL_float_decompress_openmp_threadblock_arg(decData, nbEle, errBound, blockSize, cmpData);

        ZCCL_float_decompress_openmp_threadblock_arg(
            decData2, nbEle, errBound, blockSize, cmpData2);
    }
    cost_end();
    normal_decom_cost = totalCost / ITERATIONS;

    cost_start();
    for (size_t i = 0; i < ITERATIONS; i++) {
        add_vectors(adddecData_normal, decData, decData2, nbEle);
    }
    cost_end();
    normal_cpt_cost = totalCost / ITERATIONS;

    cost_start();
    for (size_t i = 0; i < ITERATIONS; i++) {
        ZCCL_float_openmp_threadblock_arg(
            addcmpData_normal, adddecData_normal, &addcmpSize_normal, errBound, nbEle, blockSize);
    }
    cost_end();
    normal_com_cost = totalCost / ITERATIONS;

    normal_total_cost = normal_decom_cost + normal_cpt_cost + normal_com_cost;
    printf(
        "Traditional DOC workflow (decompression+operation+compression) performance: time: %f s, throughput: %f GBps\n",
        normal_total_cost,
        nbEle * sizeof(float) / normal_total_cost / 1000 / 1000 / 1000);

    cost_start();

    for (size_t i = 0; i < ITERATIONS; i++) {
        ZCCL_float_homomophic_add_openmp_threadblock(
            addcmpData, &addcmpSize, nbEle, errBound, blockSize, cmpData, cmpData2);
    }
    cost_end();
    homo_total_cost = totalCost / ITERATIONS;
    printf("hZ-dynamic performance: time: %f s, throughput: %f GBps\n",
        homo_total_cost,
        nbEle * sizeof(float) / homo_total_cost / 1000 / 1000 / 1000);
    printf("hZ-dynamic speedup: %0.2fX\n", normal_total_cost / homo_total_cost);
    cost_start();
    for (size_t i = 0; i < ITERATIONS; i++) {
        ZCCL_float_decompress_openmp_threadblock_arg(
            adddecData, nbEle, errBound, blockSize, addcmpData);
    }
    cost_end();

    cost_start();
    for (size_t i = 0; i < ITERATIONS; i++) {
        ZCCL_float_decompress_openmp_threadblock_arg(
            adddecData_2, nbEle, errBound, blockSize, addcmpData_normal);
    }
    cost_end();

    printf("Traditioanl DOC workflow quality:\n");
    error_evaluation(addoriData, adddecData_2, nbEle, addcmpSize_normal);
    printf("hZ-dynamic quality:\n");
    error_evaluation(addoriData, adddecData, nbEle, addcmpSize);

    printf("\n");

    free(oriData);
    free(cmpData);
    free(decData);
    free(oriData2);
    free(cmpData2);
    free(decData2);
    free(addcmpData);
    free(addoriData);
    free(adddecData);
    free(adddecData_2);
    free(adddecData_normal);

    return 0;
}
