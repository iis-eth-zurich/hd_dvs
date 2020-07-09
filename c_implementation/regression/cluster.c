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

#include "rt/rt_api.h"
#include <stdio.h>
#include "cluster.h"


void cluster_entry()
/*
main function executing quadratic features calculation and regression inference
*/
{
	unsigned int i;

	//precomputed features extracted from DVS sequences
	double feat[6];
    feat[0] = 0.5468;
    feat[1] = -0.1527;
    feat[2] = -0.2559;
    feat[3] = -0.3322;
    feat[4] = -0.2254;
    feat[5] = 0.3626;

	double matrix_multiplier[22];

	for (i = 0; i < 22; i++)
	{
		matrix_multiplier[i] = (double)i; //necessary initialization of regression weights from pretrained model
	}
	
	double transformed_features[22];
	double result = 0.0;

	transformed_features[0] = 1;
	transformed_features[1] = feat[0] * feat[0];
	transformed_features[2] = feat[1] * feat[1];
	transformed_features[3] = feat[2] * feat[2];
	transformed_features[4] = feat[3] * feat[3];
	transformed_features[5] = feat[4] * feat[4];
	transformed_features[6] = feat[5] * feat[5];
	transformed_features[7] = feat[0] * feat[1];
	transformed_features[8] = feat[0] * feat[2];
	transformed_features[9] = feat[0] * feat[3];
	transformed_features[10] = feat[0] * feat[4];
	transformed_features[11] = feat[0] * feat[5];
	transformed_features[12] = feat[1] * feat[2];
	transformed_features[13] = feat[1] * feat[3];
	transformed_features[14] = feat[1] * feat[4];
	transformed_features[15] = feat[1] * feat[5];
	transformed_features[16] = feat[2] * feat[3];
	transformed_features[17] = feat[2] * feat[4];
	transformed_features[18] = feat[2] * feat[5];
	transformed_features[19] = feat[3] * feat[4];
	transformed_features[20] = feat[3] * feat[5];
	transformed_features[21] = feat[4] * feat[5];

	for (i = 0; i < 22; i++)
	{
		result = result + transformed_features[i] * matrix_multiplier[i];
	}

    return 0;
}
