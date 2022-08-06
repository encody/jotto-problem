#!/usr/bin/env bash

RUSTFLAGS="-C target-cpu=native" cargo build --release && time ./target/release/jotto-problem

