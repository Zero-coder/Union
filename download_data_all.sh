#!/bin/bash

mkdir -p dataset

echo "Dataset download hooks for Union."
echo "Mirror the fecam benchmark layout under ./dataset before full reproduction:"
echo "  ETT-small, electricity, exchange_rate, weather, illness"
echo "  UEA/UCR classification archives"
echo "  MSL, PSM, SMAP, SMD, SWAT anomaly datasets"
echo
echo "This scaffold intentionally does not auto-download large benchmark files yet."
