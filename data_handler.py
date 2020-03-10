# Download and import the MIT 6.S191 package
import mitdeeplearning as mdl
import numpy as np
import random
import contextlib
import io
import sys
import nltk
#
# class DataHandler(object):
#     def __init__(self, seq_length, batch_size):
#         songs = mdl.lab1.load_training_data()
#         self.batch_size = batch_size
#         self.seq_length = seq_length
#
#     def get_batch(self, training=True):
#
#         if training:
#           random.shuffle(train_songs)
#           # Join our list of song strings into a single string containing all songs
#           songs_joined = "\n\n".join(train_songs)
#         else:
#           random.shuffle(test_songs)
#           # Join our list of song strings into a single string containing all songs
#           songs_joined = "\n\n".join(test_songs)
#
#         def vectorize_string(string):
#           return np.array([char2idx[char] for char in string])
#
#         vectorized_songs = vectorize_string(songs_joined)
#
#         assert isinstance(vectorized_songs,
#                           np.ndarray), "returned result should be a numpy array"
#         # the length of the vectorized songs string
#         n = vectorized_songs.shape[0] - 1
#         # randomly choose the starting indices for the examples in the training batch
#         song_start_idcs = np.where(vectorized_songs == char2idx['X'])[0]
#         song_start_idcs = np.array(
#           [idx for idx in song_start_idcs if idx + self.seq_length < n])
#         idx = np.random.choice(len(song_start_idcs), self.batch_size)
#         idx = song_start_idcs[idx]
#
#         '''TODO: construct a list of input sequences for the training batch'''
#         input_batch = [vectorized_songs[i: i + self.seq_length] for i in idx]
#         '''TODO: construct a list of output sequences for the training batch'''
#         output_batch = [vectorized_songs[i + 1: i + self.seq_length + 1] for i in
#                         idx]
#
#         # x_batch, y_batch provide the true inputs and targets for network training
#         x_batch = np.reshape(input_batch, [self.batch_size, self.seq_length])
#         y_batch = np.reshape(output_batch, [self.batch_size, self.seq_length])
#         return x_batch, y_batch


# Download the dataset
songs = mdl.lab1.load_training_data()
songs[21] = songs[21] + songs[22]

#remove titles, numbers and some weeird entries
start_char = 'Z'
songs = [song[song.find(start_char):]  for song in songs if song[0] == 'X' and song.find(start_char) and song[-1] == '!']
# Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs)

# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
vocab_size = len(vocab)
print("There are", len(vocab), "unique characters in the dataset")

### Define numerical representation of text ###

# Create a mapping from character to unique index.
# For example, to get the index of the character "d",
#   we can evaluate `char2idx["d"]`.
char2idx = {u:i for i, u in enumerate(vocab)}

# Create a mapping from indices to characters. This is
#   the inverse of char2idx and allows us to convert back
#   from unique index to the character in our vocabulary.
idx2char = np.array(vocab)

song_lengths = [len(song) for song in songs]
print('min song length is {}'.format(min(song_lengths)))
print('max song length is {}'.format(max(song_lengths)))
print('mean song length is {}'.format(np.mean(np.array((song_lengths)))))

for i in range(20):
  print(songs[i])


train_songs = songs[:700]
test_songs = songs[701:]

def get_batch(seq_length, batch_size, training=True, start_with_song=True):

  if training:
    random.shuffle(train_songs)
    # Join our list of song strings into a single string containing all songs
    songs_joined = "\n\n" + "\n\n".join(train_songs) + "\n\n"
  else:
    random.shuffle(test_songs)
    # Join our list of song strings into a single string containing all songs
    songs_joined = "\n\n" + "\n\n".join(test_songs) + "\n\n"

  def vectorize_string(string):
    return np.array([ char2idx[char] for char in string])

  vectorized_songs = vectorize_string(songs_joined)

  assert isinstance(vectorized_songs, np.ndarray), "returned result should be a numpy array"
  # the length of the vectorized songs string
  n = vectorized_songs.shape[0] - 1
  # randomly choose the starting indices for the examples in the training batch
  if start_with_song:
    song_start_idcs = np.where(vectorized_songs == char2idx[start_char])[0]
    song_start_idcs = np.array(
      [idx - 2 for idx in song_start_idcs if idx - 2 + seq_length < n])
    idx = np.random.choice(len(song_start_idcs), batch_size)
    idx = song_start_idcs[idx]
  else:
    idx = np.random.choice(n - seq_length, batch_size)

  '''TODO: construct a list of input sequences for the training batch'''
  input_batch = [vectorized_songs[i: i + seq_length] for i in idx]
  '''TODO: construct a list of output sequences for the training batch'''
  output_batch = [vectorized_songs[i + 1: i + seq_length + 1] for i in
                  idx]

  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch = np.reshape(input_batch, [batch_size, seq_length])
  y_batch = np.reshape(output_batch, [batch_size, seq_length])
  return x_batch, y_batch



@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

def extract_songs(samples):
  songs_out = []
  for i in range(samples.shape[0]):
    sample_string =''.join(idx2char[np.array(samples[i, ...])])
    # with nostdout():
    songs =  mdl.lab1.extract_song_snippet(sample_string)
    songs_out.append(songs)
    songs = [ 'X: 1\nT: Genned song\n' + song for song in songs]

    for ii, song in enumerate(songs):

      print(song)
      # abc_file = mdl.lab1.save_song_to_abc(song, filename="tmp_".format(ii))
      # path_to_tool = os.path.join(os.getcwd(), 'bin', 'abc2wav')
      # # cmd = "{} {}".format(path_to_tool, 'tmp.abc')
      # ret =  os.system('./abc2wav {}'.format('tmp.abc'))
      # if ret == 0:
      #   print('successful')

  return songs_out


def get_bleu_score(genned_songs):
    bleu_score = 0
    songs_list = [list(song) for song in songs]
    for genned_song in genned_songs:
        if len(genned_song) > 0:
            bleu_score += nltk.translate.bleu_score.sentence_bleu(songs_list, list(genned_song[0]), weights=[1])

    bleu_score /= len(genned_songs)
    return bleu_score