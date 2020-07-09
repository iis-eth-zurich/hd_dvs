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
#include "fann.h"
#include "fann_conf.h"


void cluster_entry()
/*
function executing the network on precomputed and previously rescaled features
*/
{
    fann_type *calc_out;
    fann_type input[6];
    
    //precomputed features --> previously rescaled features, possible to add rescaling
    input[0] = 472;
    input[1] = -109;
    input[2] = 52;
    input[3] = -608;
    input[4] = 201;
    input[5] = 94;

    calc_out = fann_run(input);

    return 0;
}
