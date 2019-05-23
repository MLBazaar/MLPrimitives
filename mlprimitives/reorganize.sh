# !/bin/bash

function reorganize() {
    cd $1
    for name in $(ls -1 *.*.json | cut -d'.' -f1 | sort | uniq); do
        mkdir -p $name
        for file in $(ls $name.*.json); do
            mv $file $name/$(echo $file | cut -d'.' -f2-)
        done
        reorganize $name
    done
    cd ..
}

reorganize $1
