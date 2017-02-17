import os
import logging
logging.basicConfig(level=getattr(logging,
                                  os.environ.get('TVB_LOG', 'WARNING')))
