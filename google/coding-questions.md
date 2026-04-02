# Coding Questions

* Log Parsing
  * "Here is a raw text file of a chat conversation `[User]... [Bot]...`. Parse it into a pandas DataFrame with columns `turn_id`, `speaker`, `text`, `latency`."
* String Manipulation
  * Regex
  * tokenization (splitting by whitespace or sentence boundaries)
  * calculating overlap between two strings.

* Practice Problems
  - Calculate per-user metrics from a conversation log
  - Compute rolling averages or sessionization
  - Join user attributes to event data and compute conditional metrics
  - Flag anomalies or outliers in a time series

## Simulate a coin flip process

```python
import random
def simulate_coin_flips(n):
    flips = [random.choice(['H', 'T']) for _ in range(n)]
    return flips
# Example usage:
n = 10
flips = simulate_coin_flips(n)
print(flips)
```

## API Simulation

***Write a function that calls this (mock) LLM API, handles rate limits/retries, and processes the JSON response.***

```python
import time
import random
import json

def api_get(prompt):
    # Simulate API call with random success/failure
    if random.random() < 0.8:  # 80% chance of success
        return {"response": f"LLM response to: {prompt}"}
    else:
        raise Exception("API rate limit exceeded")

def robust_call_llm_api(prompt, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            response = api_get(prompt)
            return response
        except Exception as e:
            print(f"Error: {e}. Retrying in {2 ** retries} seconds...")
            time.sleep(2 ** retries)  # Exponential backoff
            retries += 1
    raise Exception("Max retries exceeded")

# Example usage:
prompt = "What is the capital of France? Answer in JSON format: {'response': <answer>}"
response = robust_call_llm_api(prompt)
print(json.dumps(response, indent=2))
```

## Random sample of size `n` from a list `x`, without replacement

```python
x = [1, 2, 3, 4, 5]

# version 1
random.shuffle(x)
sample = x[:n]

# version 2
sample = random.sample(population=x, k=n)
```

## Random sample of size `n` from a list `x`, with replacement
```python
x = [1, 2, 3, 4, 5]
sample = [x[random.randint(0, len(x))] for _ in range n]
sample = random.choices(population=x, k=n)
```


## Create a random list of numbers of length `n`

```python
# floats
x = [random.random() for _ in range(n)]

# integers between a and b (NB: b is inclusive)
x = [random.randint(a, b) for _ in range(n)]
```

## Implement Ranking Functions

### Competitive Ranking (1224)

### Dense Ranking (1223)

### Ordinal Ranking (1234)

### Fractional Ranking (1.5, 1.5, 3)


## Calculate the mean, median, mode, variance, and standard deviation of a list of numbers.

## Calculate the Pearson correlation coefficient between two lists of numbers.

## Implement a function to perform matrix multiplication.

## Implement a function to find the nth Fibonacci number using dynamic programming.

## Calculate the $n$th quantile of a list of numbers.

## Implement k-means clustering from scratch.

## Implement k-nearest neighbors (k-NN) algorithm from scratch.

## Implement simple linear regression from scratch.

## Calculate bootstrap standard errors for the mean of a list of numbers.

## Calculate the bootstrap confidence interval for the mean of a list of numbers.

## (Bootstrap Hypothesis Testing) Implement a function to perform a bootstrap hypothesis test for the difference in means between two lists of numbers. 

### Confidence Interval Version

### p-value Version