.PHONY: test eval demo

test:
	python3 -m pytest -q

eval:
	python3 -m cite_or_refuse.cli eval

demo:
	python3 -m cite_or_refuse.cli ask "How much storage does the Free plan include?"
