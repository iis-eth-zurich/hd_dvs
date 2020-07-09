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

#include "lib.h"


void insertandpop(int *elements_, int *indexes_, int toinsert, int position, int index)
/*
adds new element into highest k elements vector and shifts the others to maintain ordering
args:
    elements_: vector containing the highest k scores (dimension depends on desired density)
    indexes_: indexes of the position of the highest k scores
    toinsert: new score to be added to elements_
    position: position inside elements_ and indexes_ where to add new score and index
    index: index corresponding to the score to add
*/
{
    int j;
    for (j=28; j>position; j--)
    {
        *(elements_ + j) = *(elements_ + (j-1));
        *(indexes_ + j) = *(indexes_ + (j-1));
    }
    *(elements_ + position) = toinsert;
    *(indexes_ + position) = index;
    return;
}

int indexes[29];
int elements[29];


void gethdv(unsigned int *vector, int *features, int *randencoder)
/*
function initializing the HD sparse vector using SKC algorithm
args:
    vector: pointer to vector to initialize
    features: pointer to features vector
    randencoder: pointer to random mapper used to compute sparse vector
*/
{
    int resulting;
    int i, j;
    int temp;

    #pragma omp parallel for shared(elements) private(i) schedule(static) num_threads(16)
    for (i = 0; i < 29; ++i)
    {
        elements[i] = -100;
    }

    #pragma omp parallel for shared(features, randencoder, elements) private(i, j, temp, resulting) schedule(static) num_threads(16)
    for (i = 0; i < 2880; i++)
    {
	    temp = 0.0;
        for (j=0; j<6; j++)
        {
            temp += (*(features + j)) * (*(randencoder + (i*6+j)));
        }
        resulting = temp;
        #pragma omp critical
        {
            for (j = 0; j < 29; j++)
            {
                if (resulting > elements[j])
                {
                    insertandpop(elements, indexes, resulting, j, i);
                    break;
                }
            }
        }
    }
    for (i = 0; i < 29; i++)
    {
        int idx = indexes[i];
        int bytepos = (int)(idx/32);
        int bitpos = (int)(idx%32);
        unsigned int insert = 1;
        insert = insert << bitpos;
        *(vector + bytepos) = *(vector + bytepos) | insert;
    }
    return;
}
