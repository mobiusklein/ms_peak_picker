test:
	py.test -v ./tests --cov=ms_peak_picker --cov-report=html --cov-report term

retest:
	py.test -v ./tests --pdb --lf

sphinx:
	cd docs && make html

dev:
	python setup.py develop
