import os
import json
import traceback
import src.fasta as fasta

from Bio import Entrez
import jellyfish
import igmat.igmat as igmat

# Set mandatory parameters
Entrez.email = 'g.maccari@toscanalifesciences.org'

class Pubmed():
  def __init__(self, cache_path, force=False):
    self._cache_path = cache_path
    self._force = force

    # Load the associative list of antibody names
    self._lut_path = os.path.join(self._cache_path, 'lut.json')
    if os.path.exists(self._lut_path):
      with open(self._lut_path, 'r') as handle:
        self._lut = json.load(handle)
    else:
      self._lut = {}

  def __del__(self):
    self.store()

  def store(self):
    '''
    Store the lut object
    '''

    with open(self._lut_path, 'w') as handle:
      json.dump(self._lut, handle, indent=2)

  def fetch(self, name, pmid):

    # Fetch one or multiple pmid sequences
    pmid = pmid if isinstance(pmid, list) else [ pmid ]
    sequence_list = []
    for id in pmid:
      sequence_list.extend(self._fetch_sequence(id))

    # Populate the result with data from the LUT
    result = {}
    lut_key = [ '{}_H'.format(name), '{}_L'.format(name) ]
    for key in lut_key:
      if key not in self._lut:
        continue

      # chain_type = key.split('_')[1]
      result[key] = self._lut[key]
        
    # Search for matches
    if len(result.keys()) < 2:

      # Annotate sequences to find antibodies
      ab_list = []
      for sequence in sequence_list:

        try:
          resultList = igmat.annotate(sequence['sequence'], 'IMGT')
        except:
          continue

        annotation = resultList[0]
        chain_type = 'H' if annotation.type == 'H' else 'L'

        # Append to ab_list
        ab_list.append({
          'name': sequence['name'],
          'sequence': sequence['sequence'],
          'annotation': annotation.sequence,
          'type': chain_type
        })

      # Parse ab match results and find the one with higher score
      ab_score = {}
      for ab in ab_list:

        ab_type = ab['type']
        for token in ab['name'].split():
          score = jellyfish.jaro_distance(name, token)
          if ab_type not in ab_score or score > ab_score[ab_type]['score']:
            ab_score[ab_type] = {
              'score': score,
              'sequence': ab['sequence'],
              'header': ab['name'],
              'type': ab['type']
            }

      # Set headers in the result list
      for ab_type in ab_score:
        if ab_score[ab_type]['score'] > 0.7:
          key = '{}_{}'.format(name, ab_type)
          if key not in result:
            result[key] = ab_score[ab_type]['header']
            self._lut[key] = ab_score[ab_type]['header']
          #result[ab_type] = ab_score[ab_type]['sequence']

    # Extract sequences according to the associated header
    ab_sequence = {}
    for key, value in result.items():

      chain_type = key.split('_')[1]
      for i in range(len(sequence_list)):
        if sequence_list[i]['name'] == value:
          ab_sequence[chain_type] = sequence_list[i]['sequence']
          break

    if len(ab_sequence) == 0:
      print('Unable to find antibody {} in sequence: {}'.format(name, ('no sequences' if len(sequence_list) == 0 else '')))
      for sequence in sequence_list:
        print(' - {}'.format(sequence['name']))

    return ab_sequence

  def _fetch_sequence(self, pmid, force=False):

    pmid_cache = os.path.join(self._cache_path, '{}.cache'.format(pmid))
    if not os.path.exists(pmid_cache):
      webenv = None
      query_key = None
      for i in range(5):
        try:
          # Perform search for all protein sequences related to the publication
          if not webenv or not query_key:
            with Entrez.elink(
              dbfrom='pubmed', 
              db='protein',
              id=str(pmid), 
              webenv=None, 
              query_key=None,
              cmd='neighbor_history'
            ) as handle:

              record = Entrez.read(handle)
              print(record)
              print(record[0]['WebEnv'])
              print(record[0].keys())
              query_key = record[0]['LinkSetDbHistory'][0]['QueryKey']
              webenv = record[0]['WebEnv']
        
          # We haven't found any match
          if not webenv or not query_key:
            raise Exception('Unable to perform elink')

          # Now fetch all data in fasta
          with Entrez.efetch(
            db="protein", 
            webenv=webenv, 
            query_key=query_key,
            rettype="fasta", 
            retmode="text") as handle:

            record = handle.read()
            print(record)
            with open(pmid_cache, 'w') as out_handle:
              out_handle.write(record)

            # All done
            break

        except urllib.error.URLError as e:
          print('HTTPError: {}. try {}/5'.format(str(e), i+1))
          time.sleep(1)
          pass

        except Exception as e:
          print('Error fetching data for {}: {}'.format(pmid, str(e)))
          print(traceback.format_exc())

          # Write to cache anyways
          with open(pmid_cache, 'w') as handle:
            handle.write('')

          # No point trying again
          break

    # Parse data
    fasta_list = []
    for name, sequence in fasta.parse(pmid_cache):
      fasta_list.append({
        'name':name,
        'sequence': sequence
      })

    return fasta_list

    # sys.exit(1)