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

/*insertandpop
adds new element into highest k elements vector and shifts the others to maintain ordering
args:
    elements_: vector containing the highest k scores (dimension depends on desired density)
    indexes_: indexes of the position of the highest k scores
    toinsert: new score to be added to elements_
    position: position inside elements_ and indexes_ where to add new score and index
    index: index corresponding to the score to add
*/
void insertandpop(int *elements, int *indexes, int toinsert, int position, int arg);

/*gethdv
function initializing the HD sparse vector using SKC algorithm
args:
    vector: pointer to vector to initialize
    features: pointer to features vector
    randencoder: pointer to random mapper used to compute sparse vector
*/
void gethdv(unsigned int *vector, int *features, int *randencoder);
