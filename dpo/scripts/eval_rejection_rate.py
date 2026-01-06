import torch
import numpy as np
import pickle
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import nltk
nltk.download('punkt')

#LOAD YOUR DPO MODEL OUTPUTS
def load_dpo_outputs(output_file):

    """
    Load the text that your DPO model already generated
    
    """

    with open(output_file, 'r') as f:

        outputs = [json.loads(line) for line in f]

    return outputs


# LOAD K-MEANS CENTROIDS
def load_kmeans_centroids(centroid_path):

    """

    Load the saved k-means centroids from k-SEMSTAMP initialization

    """

    with open(centroid_path, 'rb') as f:

        centroids = pickle.load(f)

    return centroids


# LOAD SENTENCE ENCODER
def load_sentence_encoder(model_name):

    """
    Load the same sentence encoder used in k-SEMSTAMP
    
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)

    model.eval()

    model.cuda()

    return tokenizer, model


def encode_sentence(sentence, tokenizer, model):

    """

    Get sentence embedding

    """

    inputs = tokenizer(

        sentence, 

        return_tensors='pt', 

        padding=True, 

        truncation=True,

        max_length=128

    ).to('cuda')

    

    with torch.no_grad():

        outputs = model(**inputs)

        # Mean pooling

        embedding = outputs.last_hidden_state.mean(dim=1)

        # Normalize

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    

    return embedding.cpu().numpy()[0]


# CLUSTER ASSIGNMENT
def assign_cluster(embedding, centroids):

    """

    Find nearest cluster centroid using cosine distance

    """

    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)


    # Compute cosine similarity to all centroids
    similarities = []

    for centroid in centroids:

        centroid_norm = centroid / np.linalg.norm(centroid)

        sim = np.dot(embedding, centroid_norm)

        similarities.append(sim)

    
    # Return index of nearest (highest similarity)
    return np.argmax(similarities)


# CHECK MARGIN CONSTRAINT

def check_margin(embedding, cluster_id, centroids, margin=0.035):

    """

    Check if embedding satisfies margin constraint

    Distance to nearest centroid must be smaller than 

    distance to second-nearest by at least 'margin'

    """

    # Normalize

    embedding = embedding / np.linalg.norm(embedding)

    

    # Compute distances to all centroids

    distances = []

    for i, centroid in enumerate(centroids):

        centroid_norm = centroid / np.linalg.norm(centroid)

        # Cosine distance = 1 - cosine similarity

        dist = 1.0 - np.dot(embedding, centroid_norm)

        distances.append(dist)

    

    # Sort distances
    sorted_dists = sorted(distances)


    # Check margin: d_second - d_first > margin
    margin_satisfied = (sorted_dists[1] - sorted_dists[0]) > margin

    return margin_satisfied


# DETERMINE VALID CLUSTERS

def get_valid_clusters(prev_cluster_id, num_clusters=8, gamma=0.25, seed_multiplier=1000):

    """

    Pseudo-randomly determine which clusters are valid at this step

    Based on previous cluster ID (just like k-SEMSTAMP)

    """

    num_valid = int(num_clusters * gamma)
    

    # Use previous cluster as seed

    seed = prev_cluster_id * seed_multiplier

    rng = np.random.RandomState(seed)


    # Randomly sample valid cluster indices

    valid_clusters = rng.choice(num_clusters, size=num_valid, replace=False)

    
    return set(valid_clusters)


# MAIN EVALUATION FUNCTION

def evaluate_rejection_rate(

    dpo_outputs,

    centroids,

    sentence_encoder_tokenizer,

    sentence_encoder_model,

    num_clusters=8,

    gamma=0.25,

    margin=0.035

):

    """

    Main function: test DPO outputs against k-SEMSTAMP constraints

    """

    total_sentences = 0

    rejected_cluster = 0

    rejected_margin = 0

    accepted = 0

    results_per_sample = []
    
    print("Evaluating rejection rate...")


    for idx, sample in enumerate(tqdm(dpo_outputs)):

        # Extract generated text

        if isinstance(sample, dict):

            text = sample.get('generated_text', sample.get('output', ''))

        else:

            text = sample

        
        # Split into sentences

        sentences = nltk.sent_tokenize(text)


        # Track previous cluster for seeding

        prev_cluster_id = 0  # Start with cluster 0

        
        sample_stats = {

            'sample_id': idx,

            'total_sentences': len(sentences),

            'accepted': 0,

            'rejected_cluster': 0,

            'rejected_margin': 0

        }

        

        for sent in sentences:

            if len(sent.strip()) < 5:  # Skip very short sentences

                continue

            
            total_sentences += 1

            

            # Encode sentence

            embedding = encode_sentence(

                sent, 

                sentence_encoder_tokenizer, 

                sentence_encoder_model

            )

            

            # Assign to nearest cluster

            cluster_id = assign_cluster(embedding, centroids)

            

            # Determine valid clusters for this position

            valid_clusters = get_valid_clusters(

                prev_cluster_id,

                num_clusters=num_clusters,

                gamma=gamma

            )

            

            # Check if cluster is valid

            if cluster_id not in valid_clusters:

                rejected_cluster += 1

                sample_stats['rejected_cluster'] += 1

                prev_cluster_id = cluster_id  # Update for next sentence

                continue

            

            # Check margin constraint

            if not check_margin(embedding, cluster_id, centroids, margin):

                rejected_margin += 1

                sample_stats['rejected_margin'] += 1

                prev_cluster_id = cluster_id

                continue

            

            # ACCEPTED!

            accepted += 1

            sample_stats['accepted'] += 1

            prev_cluster_id = cluster_id

        results_per_sample.append(sample_stats)


    # Calculate statistics

    total_rejected = rejected_cluster + rejected_margin

    rejection_rate = total_rejected / total_sentences if total_sentences > 0 else 0

    acceptance_rate = accepted / total_sentences if total_sentences > 0 else 0

    effective_attempts = total_sentences / accepted if accepted > 0 else float('inf')


    results = {

        'total_sentences': total_sentences,

        'accepted': accepted,

        'rejected_cluster': rejected_cluster,

        'rejected_margin': rejected_margin,

        'total_rejected': total_rejected,

        'rejection_rate': rejection_rate,

        'acceptance_rate': acceptance_rate,

        'effective_attempts': effective_attempts,

        'per_sample_stats': results_per_sample

    }


    return results


# RUN THE EVALUATION

if __name__ == "__main__":

    import argparse

    

    parser = argparse.ArgumentParser()

    parser.add_argument('--dpo_outputs', type=str, required=True,

                        help='Path to DPO model outputs (jsonl file)')

    parser.add_argument('--centroids', type=str, required=True,

                        help='Path to k-means centroids (pickle file)')

    parser.add_argument('--encoder_model', type=str, 

                        default='sentence-transformers/all-mpnet-base-v1',

                        help='Sentence encoder model name')

    parser.add_argument('--num_clusters', type=int, default=8)

    parser.add_argument('--gamma', type=float, default=0.25,

                        help='Valid region ratio')

    parser.add_argument('--margin', type=float, default=0.035,

                        help='Margin constraint threshold')

    parser.add_argument('--output_file', type=str, default='rejection_rate_results.json',

                        help='Where to save results')

    

    args = parser.parse_args()

    

    print("Loading components...")

    

    # Load DPO outputs

    print(f"Loading DPO outputs from: {args.dpo_outputs}")

    dpo_outputs = load_dpo_outputs(args.dpo_outputs)

    print(f"Loaded {len(dpo_outputs)} samples")

    

    # Load centroids

    print(f"Loading centroids from: {args.centroids}")

    centroids = load_kmeans_centroids(args.centroids)

    print(f"Loaded {len(centroids)} centroids")

    

    # Load sentence encoder

    print(f"Loading sentence encoder: {args.encoder_model}")

    tokenizer, model = load_sentence_encoder(args.encoder_model)

    print("Encoder loaded")

    

    # Run evaluation

    results = evaluate_rejection_rate(

        dpo_outputs=dpo_outputs,

        centroids=centroids,

        sentence_encoder_tokenizer=tokenizer,

        sentence_encoder_model=model,

        num_clusters=args.num_clusters,

        gamma=args.gamma,

        margin=args.margin

    )

    
    # Print results

    print("\n" + "="*50)

    print("RESULTS")

    print("="*50)

    print(f"Total sentences evaluated: {results['total_sentences']}")

    print(f"Accepted: {results['accepted']} ({results['acceptance_rate']:.2%})")

    print(f"Rejected (cluster): {results['rejected_cluster']}")

    print(f"Rejected (margin): {results['rejected_margin']}")

    print(f"Total rejected: {results['total_rejected']} ({results['rejection_rate']:.2%})")

    print(f"\nEffective attempts per sentence: {results['effective_attempts']:.2f}")

    print(f"k-SEMSTAMP baseline: 13.3 attempts")

    print(f"Improvement: {13.3 / results['effective_attempts']:.1f}×")

    print("="*50)

    

    # Save detailed results

    with open(args.output_file, 'w') as f:

        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {args.output_file}")
