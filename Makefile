PY ?= python

.PHONY: all setup data train ips eval figures demo test clean

all: train ips eval figures test

setup:
	pip install -r requirements.txt

data:
	bash scripts/fetch_data.sh

train:
	$(PY) src/train.py --dataset nsl-kdd unsw-nb15

ips:
	$(PY) src/ips.py --dataset nsl-kdd   --replay 999999 --target-fpr 0.01
	$(PY) src/ips.py --dataset unsw-nb15 --replay 999999 --target-fpr 0.01

eval:
	$(PY) src/audit_signatures.py
	$(PY) src/novelty.py
	$(PY) src/calibration.py
	$(PY) src/robustness.py --dataset nsl-kdd --n-seeds 5
	$(PY) src/report.py

figures:
	$(PY) src/figures.py

demo:
	$(PY) scripts/make_demo_gif.py

engine:
	$(PY) engine-rs/export_model.py --dataset nsl-kdd
	$(PY) engine-rs/export_model.py --dataset unsw-nb15
	cd engine-rs && cargo build --release

parity:
	$(PY) engine-rs/parity.py --dataset nsl-kdd
	$(PY) engine-rs/parity.py --dataset unsw-nb15

native:
	$(PY) engine-native/codegen.py --dataset nsl-kdd
	cd engine-native && cargo build --release
	engine-native/target/release/nidps-engine-native

test:
	$(PY) -m pytest tests/ -q

clean:
	rm -f results/figures/*.png results/figures/*.gif
