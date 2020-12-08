# README.md - scripts and notebooks for the TESS LOFAR paper

### Order of Execution

`scripts.py` - library of functions called for data analysis.

`paper_version.py` - reads in all of the data and analyses it with `stella`

`paper_simultaneous.py` - does simultaneous plots, loads flare predictions just from `paper_version.py`.

`collect_paper.py` - generate tables from files produced in `paper_version.py` or `paper_simultaneous.py`.

`vetting_plots.py` - the output from `stella` includes many false positives. These are now hard-coded in `scripts.py`. Generate plots in `results/simultaneous/vetting` to flag these by eye. Then run `paper_simultaneous.py` and `collect_paper.py` again.

---------------

`automatic.ipynb` - do the full analysis on a single target in a notebook format.

`paper_plots.ipynb` - prepare the publication plots.