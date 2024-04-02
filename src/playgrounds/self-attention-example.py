import torch
import torch.nn.functional as F


BATCH_SIZE = 4
SEQUENCE_LENGTH = 8
EMBEDDING_SIZE = 16
HEAD_SIZE = 4

# [T, T]
mask = torch.tril(torch.ones((SEQUENCE_LENGTH, SEQUENCE_LENGTH)))

# [D, H]
query_W = torch.randn((EMBEDDING_SIZE, HEAD_SIZE))
key_W = torch.randn((EMBEDDING_SIZE, HEAD_SIZE))
value_W = torch.randn((EMBEDDING_SIZE, HEAD_SIZE))

def compute_attention_matrix(X):
	# [B, T, H]
	queries = torch.einsum("btd,dh->bth", X, query_W)
	keys = torch.einsum("btd,dh->bth", X, key_W)
	values = torch.einsum("btd,dh->bth", X, value_W)

	# queries = X @ query_W
	# keys = X @ key_W
	# values = X @ value_W

	# [B, T, T]
	attention_scores = torch.einsum("bqh,bkh->bqk", queries, keys) * (HEAD_SIZE ** -0.5)
	# attention_scores = queries @ keys.transpose(-2, -1) * (HEAD_SIZE ** -0.5)
	masked_attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
	attention_probabilities = F.softmax(masked_attention_scores, dim=-1)

	# [B, T, H]
	attention_outputs = torch.einsum("bqk,bkh->bqh", attention_probabilities, values)
	# attention_outputs = attention_probabilities @ values

	return attention_outputs


def main():
	X = torch.randn((BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE))
	attention_outputs = compute_attention_matrix(X)
	print(attention_outputs.shape)


if __name__ == "__main__":
	main()
