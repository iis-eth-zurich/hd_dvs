/*----------------------------------------------------------------------------*
 * Copyright (C) 2020 ETH Zurich, Switzerland                                 *
 * SPDX-License-Identifier: GPL-3.0-only                                      *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU General Public License as published by       *
 * the Free Software Foundation, either version 3 of the License, or          *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU General Public License for more details.                               *
 *                                                                            *
 * You should have received a copy of the GNU General Public License          *
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.     *
 *                                                                            *
 * Author:  Edoardo Mello Rella <edoardom@student.ethz.ch>                    *
 *----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <pulp.h>
#include "cluster.h"



#define MASTER_STACK_SIZE 64000
#define SLAVE_STACK_SIZE 128

int main ()
{
	unsigned int a, b;

	int freq=100000000; //most efficient frequency
    
	rt_freq_set(RT_FREQ_DOMAIN_FC, freq);

	rt_freq_set(RT_FREQ_DOMAIN_CL, freq);

    rt_cluster_mount(1, 0, 0, NULL);

    void *stacks = rt_alloc(RT_ALLOC_CL_DATA, SLAVE_STACK_SIZE*(rt_nb_pe()-1) + MASTER_STACK_SIZE);
    if (stacks == NULL)
        return -1;

    printf("Start!\n");

    a = rt_time_get_us();

    rt_cluster_call(NULL, 0, cluster_entry, NULL, stacks, MASTER_STACK_SIZE, SLAVE_STACK_SIZE, 0, NULL);    

    b = rt_time_get_us();

    printf("Stop!\n");
    printf("timing %u\n", (b-a)/1000);

    rt_cluster_mount(0, 0, 0, NULL);

    return 0;
}
