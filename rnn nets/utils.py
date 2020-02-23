# Created by Luis Alejandro (alejand@umich.edu)
import unicodedata


def unicode_to_ascii(s: str) -> str:
  """
  Converts string from unicode to ascii

  :param s: Unicode string
  :return: ASCII string
  """
  return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
