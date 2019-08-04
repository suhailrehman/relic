#!/bin/bash

find . -name '*.csv' | xargs -i sh -c 'cut -d , -f2- {} | sponge {}'
