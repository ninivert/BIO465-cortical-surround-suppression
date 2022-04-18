def setup():
	import logging
	logging.basicConfig()
	logger = logging.getLogger('')

	import pathlib
	pathbase = pathlib.Path(__file__).parent

	try:
		import matplotlib as mpl
		import matplotlib.pyplot as plt  # needs this to access fontManager

		# Attempt to import standard TeX font
		for fontpath in (pathbase / 'lm2.004otf').glob('*.otf'):
			mpl.font_manager.fontManager.addfont(str(fontpath))  # matplotlib doesn't like non-string paths
	except Exception as e:
		logger.error(f'could not setup fonts : {e}')

	import matplotlib.pyplot as plt
	plt.style.use(pathbase / 'tufte.mplstyle')