# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import traceback
def opt_setter(d, key, value):
    if len(key) == 1:
        original_value = d.get(key[0])
        d[key[0]] = value
        print(f'Value {key[0]} set from {original_value} as {value}!')
        return
    if d.get(key[0]) is None:
        d[key[0]] = dict()
        return opt_setter(d[key[0]], key[1:], value)
    else:
        return opt_setter(d[key[0]], key[1:], value)

def key_dealer(s):
    '''Insert s without opt_'''
    return s.split('__')

def extract_parser(unparsed, d):
    if unparsed == []:
        return
    assert unparsed[0][:2] == '--'
    if '=' in unparsed[0]:
        key, value = unparsed[0].split('=')
    else:
        key = unparsed[0]
        value = unparsed[1]
    try:
        value = eval(value)
    except:
        value = str(value)
    k = key_dealer(key[2:])
    # print(f'## Set opt {k} as {value}...')
    try:
        opt_setter(d, k, value)
    except:
        print(f'## Set opt {k} as {value} FAILED...')
        traceback.print_exc()
    if '=' in unparsed[0]:
        return extract_parser(unparsed[1:], d)
    return extract_parser(unparsed[2:], d)

def printopt(opt, level=0):
    for k, it in opt.items():
        print('\t'*level + f'{k}:',end=' ')
        if not isinstance(it, dict):
            print(it)
        else:
            print()
            printopt(it, level+1)
        