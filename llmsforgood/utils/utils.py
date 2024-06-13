import IPython
import logging


logger = logging.getLogger(__name__)


def get_dbutils():
    return IPython.get_ipython().user_ns["dbutils"]


def get_spark():
    return IPython.get_ipython().user_ns["spark"]


def display(*args, **kwargs):
    return IPython.get_ipython().user_ns["display"](*args, **kwargs)


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("sh.command").setLevel(logging.ERROR)
