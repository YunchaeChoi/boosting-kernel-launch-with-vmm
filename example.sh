#!/bin/bash
TRIAL=0
CORRECT=0

for i in {1..10}
do
    RT=$(./runtime)
    DRV=$(./fixed_low_level_3mm.exe)
    echo ${RT}
    echo ${DRV}
    if (( $(echo "$RT > $DRV" |bc -l) )); then
        CORRECT=$((CORRECT+1))
        echo "hypothesis correct"
    else
        echo "hypothesis wrong"
    fi
    TRIAL=$((TRIAL+1))
done

echo $CORRECT" / "$TRIAL

PROPORTION=$(($CORRECT / $TRIAL))

exit 0
