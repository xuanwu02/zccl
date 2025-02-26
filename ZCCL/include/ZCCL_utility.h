/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *  @author Sheng Di
 *  @date Feb, 2022
 */

#ifndef _ZCCL_UTILITY_H
#define _ZCCL_UTILITY_H

#include "ZCCL.h"

#ifdef __cplusplus
extern "C" {
#endif

//sihuan added: use a assistant struct to do sorting and swap that are easy to implement: should
//consider optimizing the performance later.
typedef struct sort_ast_particle{
	int64_t id;
	float var[6];
} sort_ast_particle;

extern struct timeval sz_costStart; /*only used for recording the cost*/
extern double sz_totalCost;

void sz_cost_start();
void sz_cost_end();
void sz_cost_end_msg(char *);

#ifdef __cplusplus
}
#endif

#endif /* ----- #ifndef _ZCCL_UTILITY_H  ----- */
