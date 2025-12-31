# early_ia_rise/_utils/__init__.py

from ._data_binning import data_binning, get_box_spec
from ._dust_extinction import calALambda
from ._numpyro_utils import (
    extract_coords_dims_from_model,
    get_recommended_chain_method,
    set_best_platform,
)
from ._plt import plot_box_spec, plt, show_kde_posterior
