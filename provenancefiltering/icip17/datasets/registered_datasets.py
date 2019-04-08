# -*- coding: utf-8 -*-

from provenancefiltering.icip17.datasets.nimble2016 import Nimble2016
from provenancefiltering.icip17.datasets.nimble2016world1m import Nimble2016World1M
from provenancefiltering.icip17.datasets.nimble2017 import Nimble2017
from provenancefiltering.icip17.datasets.nimble2017world1m import Nimble2017World1M
from provenancefiltering.icip17.datasets.oxford100k import Oxford100k
from provenancefiltering.icip17.datasets.unicamp100k import Unicamp100k


registered_datasets = {0: Nimble2016,
                       1: Nimble2016World1M,
                       2: Nimble2017,
                       3: Nimble2017World1M,
                       4: Oxford100k,
                       5: Unicamp100k,
                       }
