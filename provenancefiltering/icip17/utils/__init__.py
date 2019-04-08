# -*- coding: utf-8 -*-

# -- common imports
import pdb

from provenancefiltering.icip17.utils.misc import RunInParallel
from provenancefiltering.icip17.utils.misc import RunInParallelWithReturn
from provenancefiltering.icip17.utils.misc import creating_csv
from provenancefiltering.icip17.utils.misc import get_interesting_samples
from provenancefiltering.icip17.utils.misc import get_time
from provenancefiltering.icip17.utils.misc import load_object

# -- common functions and constants
from provenancefiltering.icip17.utils.misc import memory_usage_resource
from provenancefiltering.icip17.utils.misc import modification_date
from provenancefiltering.icip17.utils.misc import mosaic
from provenancefiltering.icip17.utils.misc import progressbar
from provenancefiltering.icip17.utils.misc import retrieve_samples
from provenancefiltering.icip17.utils.misc import safe_create_dir
from provenancefiltering.icip17.utils.misc import save_interesting_samples
from provenancefiltering.icip17.utils.misc import save_object
from provenancefiltering.icip17.utils.misc import total_time_elapsed

from provenancefiltering.icip17.utils.constants import color_space_dict
from provenancefiltering.icip17.utils.constants import N_JOBS
from provenancefiltering.icip17.utils.constants import CONST
from provenancefiltering.icip17.utils.constants import DESCRIPTOR_SIZE
from provenancefiltering.icip17.utils.constants import DESCRIPTOR_TYPE
from provenancefiltering.icip17.utils.constants import DB_FEATS_BATCH_SIZE
from provenancefiltering.icip17.utils.constants import MATPLOTLIB_MARKERS
from provenancefiltering.icip17.utils.constants import PROJECT_PATH
from provenancefiltering.icip17.utils.constants import UTILS_PATH
from provenancefiltering.icip17.utils.constants import WITH_INDEX
from provenancefiltering.icip17.utils.constants import WITH_FEAT_INDEX
from provenancefiltering.icip17.utils.constants import NC2016_MANIPULATION
from provenancefiltering.icip17.utils.constants import NC2016_REMOVAL
from provenancefiltering.icip17.utils.constants import NC2016_SPLICE
from provenancefiltering.icip17.utils.constants import NC2016_FILTERED_SPLICE
from provenancefiltering.icip17.utils.constants import NIMBLE_ANNOTATION_PATH
from provenancefiltering.icip17.utils.constants import NC2017_Dev1_Beta4
from provenancefiltering.icip17.utils.constants import NC2017_Dev1_Beta4_gt
