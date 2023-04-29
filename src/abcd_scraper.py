#!/usr/bin/env python

import os
import argparse
import time
import urllib.request

from bs4 import BeautifulSoup
import pandas

from src.pubmed import Pubmed

# Get script path
file_path  = os.path.split(__file__)[0]

# Initialize the pubmed object
pubmed = Pubmed(os.path.join(file_path, 'cache'))

def url_generator():

  alphabet = 'AA'
  suffix = 'ABCD_'
  counter = 1
  while True:
    value = '{}{}{:03d}'.format(suffix,alphabet,counter)
    if counter == 999:
      counter = 1

      if alphabet == 'ZZ':
        raise Exception('End of sequence')

      if alphabet[1] == 'Z':
        alphabet = '{0}A'.format(
          chr(ord(alphabet[0]) + 1)
        )
      else:
        alphabet = '{0}{1}'.format(
          alphabet[0],
          chr(ord(alphabet[1]) + 1),
        )
    else:
      # Increment counter
      counter += 1

    yield value
  
def scrape_page(id):

  try:

    # Generate the URL
    url = 'https://web.expasy.org/abcd/{}'.format(id)
    
    # Check for cache data
    cache_path = os.path.join(file_path, 'cache', '{}.cache'.format(id))
    if not os.path.exists(cache_path):

      # Fetch page
      request = urllib.request.Request(url)
      with urllib.request.urlopen(request) as response:
        html = response.read()

      # No result
      if not html:
        raise Exception('Unable to fetch data')

      # Write to cache
      with open(cache_path, 'wb') as handle:
        handle.write(html)
    else:

      # Read previous cache
      with open(cache_path, 'rb') as handle:
        html = handle.read()

    # Parse page
    soup = BeautifulSoup(html,features="html.parser")

    # Search for main container
    main = soup.find("div", {"class": "main-elmt"})
    if not main:
      raise Exception('Unable to find main element')

    # Extract data from table
    result = {}
    for row in main.findAll('tr'):
      header = row.find('th')
      data = row.find('td')

      if not header or not data:
        continue

      # Extract any eventual URL
      href_list = [ a['href'] for a in data.findAll('a') ]

      # Parse entry
      header = header.text.lower().replace(' ', '_')
      data = data.text
      result[header] = href_list if len(href_list) > 0 else data

    # Extract data from pubmedid
    if 'publications' in result:
      pmid_list = []
      for publication_url in result['publications']:
        if not publication_url.startswith('https://www.ncbi.nlm.nih.gov/pubmed/'):
          continue

        pmid = publication_url.replace('https://www.ncbi.nlm.nih.gov/pubmed/', '')
        pmid_list.append(pmid[:8])

      # Try to fetch data
      ab_sequence = pubmed.fetch(result['antibody_name'], pmid_list)
      result['antibody_VH'] = ab_sequence.get('H', '')
      result['antibody_VL'] = ab_sequence.get('L', '')

    return result
  except Exception as e:
    print(str(e))
    return None
  except urllib.error.URLError as e:
    print('URL {} does not exists'.format(url))
    return None

def __generate_output(path):
  ''' 
  Check for the existence of the output file and eventually 
  append a sequential number to avoid duplicates
  '''
  counter = 1
  result_path = path
  while True:
    if not os.path.exists(result_path):
      break

    dir_path = os.path.dirname(path)
    base_path = os.path.basename(path)
    base_path = '{}_{}.{}'.format(
      base_path.split('.')[0],
      counter,
      base_path.split('.')[1]
    )
    result_path = os.path.join(dir_path, base_path)
    counter += 1
    

  return result_path


if __name__ == "__main__":

  # Initialize the argument parser
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-o', '--output', help='output file', dest='output', type=str, required=True)

  # Parse arguments
  args = argparser.parse_args()

  # Create cache directory
  cache_path = os.path.join(file_path, 'cache')
  if not os.path.exists(cache_path):
    os.mkdir(cache_path)

  # This is a list of headers
  header_list = [
    'id',
    'target_type', 
    'target_link', 
    'target_name', 
    'epitope', 
    'antibody_name', 
    'antibody_synonyms', 
    'applications', 
    'cross-references', 
    'publications',
    'antibody_VH',
    'antibody_VL'
  ]

  # Generate a new file name if already existing
  output_path = __generate_output(args.output)

  # Scrape each page from the database 
  counter = 0
  result = []
  for id in url_generator():

    # Generate the final url
    print('Fetching id {}'.format(id))

    # Scrape page
    data = scrape_page(id)
    if data is not None:

      # Append id & store
      data['id'] = id
      result.append(data)

      # Append to current output file
      with open(output_path, 'a') as handle:
        line = ''
        for header in header_list:
          line += '\t' if len(line) > 0 else ''
          line += str(data.get(header, ''))

        if counter == 0:
          handle.write('\t'.join(header_list) + '\n')
          
        handle.write(line + '\n')


    # Increment counter
    counter += 1

  # Store results
  result = pandas.DataFrame.from_dict(result)
  result.to_csv(args.output, sep='\t', index=False)

  # Done
  print('Scraped {} pages. Number of extracted entries: {}'.format(counter, len(result)))

  # Make sure to store the lut
  pubmed.store()