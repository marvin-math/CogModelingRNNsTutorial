import pandas as pd
import numpy as np
from scipy.optimize import minimize
import bandits
from rnn_utils import DatasetRNN, compute_log_likelihood


##########################################
# SETTINGS: Choose optimization mode
##########################################
# Set to True to optimize parameters per participant,
# or False to optimize once for the entire dataset.
OPTIMIZE_PER_PARTICIPANT = False

##########################################
# LOAD DATA
##########################################
df = pd.read_csv('/Users/marvinmathony/PycharmProjects/IntrinsicMot/data/human_data.csv')
df['state'] = df.index % 8800  # Ensure state values are within range

# Fixed parameters for the model (used both in optimization and later simulations)
fixed_params = {
    'beta': 1,      # initial beta value (if needed)
    'gamma': 2,     # initial gamma value (if needed)
    'n_states': 8800
}
initial_params = [1, 2]  # Starting values for [beta, gamma]
bounds = [(0, 10), (0, 10)]  # Bounds for beta and gamma

##########################################
# NEGATIVE LOG-LIKELIHOOD FUNCTION
##########################################
def negative_log_likelihood_bandit(opt_params, data, fixed_params):
    beta, gamma = opt_params  # Optimized parameters
    n_states = fixed_params['n_states']
    # Create a new model instance with these parameters
    model = bandits.HybridAgent_opt(n_states=n_states, beta=beta, gamma=gamma)
    print('number of states: ', model.n_states)
    print('beta: ', model.beta)
    print('gamma: ', model.gamma)
    nll = 0

    # Loop over trials in the provided data
    for _, row in data.iterrows():
        choice = int(row['choice']) - 1  # Convert human choice from 1,2 to 0,1
        reward = row['reward']
        state_in_df = row['state']
        # Get model’s probability for choosing action 0 (and action 1)
        action_prob_0 = model.get_choice_probs(state_in_df)
        action_prob_1 = 1 - action_prob_0
        actions_probs = np.array([action_prob_0, action_prob_1])
        action_prob = actions_probs[choice]

        nll -= np.log(action_prob + 1e-10)
        model.update(choice, reward, state_in_df)

    return nll

##########################################
# OPTIMIZATION
##########################################
if OPTIMIZE_PER_PARTICIPANT:
    # Optimize separately for each participant.
    results_per_participant = {}
    for subject, subject_data in df.groupby('subject'):
        print(f"\nOptimizing for subject {subject} ...")
        result = minimize(
            negative_log_likelihood_bandit,
            initial_params,
            args=(subject_data, fixed_params),
            bounds=bounds,
            method='L-BFGS-B'
        )
        results_per_participant[subject] = result
        print(f"Subject: {subject}")
        print(f"  Optimized parameters: {result.x}")
        print(f"  Final negative log-likelihood: {result.fun}\n")
else:
    # Optimize once for the entire dataset.
    print("\nOptimizing globally for the entire dataset ...")
    global_result = minimize(
        negative_log_likelihood_bandit,
        initial_params,
        args=(df, fixed_params),
        bounds=bounds,
        method='L-BFGS-B'
    )
    print("Global optimization result:")
    print(f"  Optimized parameters: {global_result.x}")
    print(f"  Final negative log-likelihood: {global_result.fun}\n")

##########################################
# CHECK ACCURACY OF THE OPTIMIZED MODEL
##########################################
accuracy_results = {}
overall_correct = 0
overall_total = 0

if OPTIMIZE_PER_PARTICIPANT:
    # Evaluate accuracy per participant using their own optimized parameters.
    for subject, subject_df in df.groupby('subject'):
        opt_result = results_per_participant[subject]
        beta_opt, gamma_opt = opt_result.x

        model_opt = bandits.HybridAgent_opt(n_states=fixed_params['n_states'],
                                            beta=beta_opt,
                                            gamma=gamma_opt)

        correct_predictions = 0
        total_predictions = 0

        for _, row in subject_df.iterrows():
            choice = int(row['choice']) - 1
            reward = row['reward']
            state_in_df = row['state']

            action_prob_0 = model_opt.get_choice_probs(state_in_df)
            action_prob_1 = 1 - action_prob_0
            predicted_choice = np.argmax([action_prob_0, action_prob_1])

            if predicted_choice == choice:
                correct_predictions += 1
            total_predictions += 1

            model_opt.update(choice, reward, state_in_df)

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        accuracy_results[subject] = accuracy
        overall_correct += correct_predictions
        overall_total += total_predictions

        print(f"Subject {subject}: Accuracy = {accuracy:.4f}")
else:
    # Evaluate accuracy over the entire dataset using the globally optimized parameters.
    beta_opt, gamma_opt = global_result.x
    model_opt = bandits.HybridAgent_opt(n_states=fixed_params['n_states'],
                                        beta=beta_opt,
                                        gamma=gamma_opt)
    correct_predictions = 0
    total_predictions = 0

    for _, row in df.iterrows():
        choice = int(row['choice']) - 1
        reward = row['reward']
        state_in_df = row['state']

        action_prob_0 = model_opt.get_choice_probs(state_in_df)
        action_prob_1 = 1 - action_prob_0
        predicted_choice = np.argmax([action_prob_0, action_prob_1])

        if predicted_choice == choice:
            correct_predictions += 1
        total_predictions += 1

        model_opt.update(choice, reward, state_in_df)

    overall_correct = correct_predictions
    overall_total = total_predictions
    accuracy_results['global'] = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Global Model Accuracy: {accuracy_results['global']:.4f}")

overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

# Optionally, create a DataFrame summarizing the accuracies.
accuracy_df = pd.DataFrame.from_dict(accuracy_results, orient='index', columns=['Accuracy'])
print("\nAccuracy Summary:")
print(accuracy_df)

##########################################
# SIMULATION OF THE BANDIT (Synthetic Dataset)
##########################################
# Set simulation parameters (these remain constant across participants for simulation)
n_trials_per_session = 10

env_params = {
    'innov_variance': 100,
    'noise_variance': 10,
    'n_actions': 2
}

# Dictionaries to store simulation outputs.
simulated_datasets_train = {}
simulated_experiments_train = {}
simulated_datasets_test = {}
simulated_experiments_test = {}

if OPTIMIZE_PER_PARTICIPANT:
    n_sessions = 20  # Adjust as needed

    # Simulate per participant using their optimized parameters.
    for subject, result in results_per_participant.items():
        agent_params = {
            'beta': result.x[0],
            'gamma': result.x[1]
        }
        # Simulate training data.
        dataset_train, experiment_list_train = bandits.create_dataset(
            agent_cls=bandits.HybridAgent,
            env_cls=bandits.GershmanBandit,
            n_trials_per_session=n_trials_per_session,
            n_sessions=n_sessions,
            agent_kwargs=agent_params,
            env_kwargs=env_params
        )
        # Simulate test data.
        dataset_test, experiment_list_test = bandits.create_dataset(
            agent_cls=bandits.HybridAgent,
            env_cls=bandits.GershmanBandit,
            n_trials_per_session=n_trials_per_session,
            n_sessions=n_sessions,
            agent_kwargs=agent_params,
            env_kwargs=env_params
        )
        simulated_datasets_train[subject] = dataset_train
        simulated_experiments_train[subject] = experiment_list_train
        simulated_datasets_test[subject] = dataset_test
        simulated_experiments_test[subject] = experiment_list_test
else:
    # Simulate once using the globally optimized parameters.
    n_sessions = 44*20  # Adjust as needed
    agent_params = {
        'beta': global_result.x[0],
        'gamma': global_result.x[1]
    }
    dataset_train, experiment_list_train = bandits.create_dataset(
        agent_cls=bandits.HybridAgent,
        env_cls=bandits.GershmanBandit,
        n_trials_per_session=n_trials_per_session,
        n_sessions=n_sessions,
        agent_kwargs=agent_params,
        env_kwargs=env_params
    )
    dataset_test, experiment_list_test = bandits.create_dataset(
        agent_cls=bandits.HybridAgent,
        env_cls=bandits.GershmanBandit,
        n_trials_per_session=n_trials_per_session,
        n_sessions=n_sessions,
        agent_kwargs=agent_params,
        env_kwargs=env_params
    )
    simulated_datasets_train['global'] = dataset_train
    simulated_experiments_train['global'] = experiment_list_train
    simulated_datasets_test['global'] = dataset_test
    simulated_experiments_test['global'] = experiment_list_test

##########################################
# CONCATENATE SIMULATED DATASETS (Row-wise)
##########################################
# Assume that each DatasetRNN object holds its data in ._xs and ._ys arrays.
# We will concatenate the underlying arrays along the episode dimension.
all_xs = np.concatenate(
    [dataset._xs for dataset in simulated_datasets_train.values()], axis=1
)
all_ys = np.concatenate(
    [dataset._ys for dataset in simulated_datasets_train.values()], axis=1
)
print("Shape of concatenated xs:", all_xs.shape)
print("Shape of concatenated ys:", all_ys.shape)

# Create one large DatasetRNN object containing data for all participants.
DatasetRNN = DatasetRNN
big_dataset = DatasetRNN(all_xs, all_ys)
print("Combined DatasetRNN created with", big_dataset._dataset_size, "episodes.")

##########################################
# CREATE DATAFRAME FROM SIMULATED EXPERIMENTS
##########################################
rows = []
# Loop over each participant's experiment list.
for subject, experiment_list in simulated_experiments_train.items():
    for session in experiment_list:
        for i in range(session.n_trials):
            rows.append({
                "subject": subject,
                "Action": session.choices[i],
                "rewards": session.rewards[i],
                "timeseries": session.timeseries[i],
                "n_trials": session.n_trials,
                "V_t": session.V_t[i],
                "TU": session.TU[i],
                "RU": session.RU[i],
            })
df = pd.DataFrame(rows)
print("\nSimulated Experiment Data (first 50 rows):")
print(df.head(50))



##### PREPROCESSING HUMAN DATA #####

#@title Load and Preprocess Human Data for RNN training

# Load data

human = True

if human:
  n_participants = 44
  df.columns = ['Participant', 'Block', 'Trial', 'mu1', 'mu2', 'Action', 'Reward', 'RT']
  df["Action"] = df["Action"] - 1


else:
  n_participants = 220

#preprocessing
df["Action"] = df["Action"].astype(int)
df["Reward"] = df["Reward"].astype(int)
#df["Participant"] = df["Participant"].astype(int)
df["Block"] = df["Block"].astype(int)
df["Trial"] = df["Trial"].astype(int)
# Constants
timesteps = 10
blocks_per_participant = 20
n_blocks = n_participants * blocks_per_participant

# Sort data
# Assuming the CSV has columns 'Participant', 'Block', and 'Trial'
df = df.sort_values(by=['Participant', 'Block', 'Trial'])

# Convert columns to numpy arrays and reshape
choices = df['Action'].to_numpy().reshape((timesteps, n_blocks), order='F')  # Use Fortran-style order
rewards = df['Reward'].to_numpy().reshape((timesteps, n_blocks), order='F')

# Check results
print(f'choices (first block): {choices[:, 0]}')  # First block's trials
print(f'rewards shape: {rewards.shape}')

print(np.isnan(choices).any())
print(np.isnan(rewards).any())



n_trials_per_session = choices.shape[0]
n_sessions = choices.shape[1]

# Stack choices and rewards for RNN compatibility
data = np.stack((choices, rewards), axis = -1)  # Shape: (timesteps, n_blocks, 2)
x_array = np.ones((n_sessions, n_trials_per_session, 2))
y_array = np.zeros((n_sessions, n_trials_per_session, 1))

for sess_i in range(n_sessions):
    # Construct previous choices and rewards
    prev_choices = np.concatenate(([0], choices[:-1, sess_i]))  # Shape: (n_trials_per_session,)
    prev_rewards = np.concatenate(([0], rewards[:-1, sess_i]))  # Shape: (n_trials_per_session,)

    # Stack features for the current session
    session_features = np.stack((prev_choices, prev_rewards), axis=1)  # Shape: (n_trials_per_session, 2)
    #print(f'session_features = {session_features}')

    # Assign to xs
    x_array[sess_i, :] = session_features

    # Assign targets (current choices) to ys
    y_array[sess_i, :, 0] = choices[:, sess_i]  # Shape: (n_trials_per_session,)
    if sess_i == 0:
      print(f'previous_choices: {prev_choices}')
      print(f'session_features = {session_features}')
      print(f'y_array: {choices[:, sess_i]}')




# Split into train and test sets (assuming 80% training split)
split_idx = int(0.8 * n_sessions)
train_xs = x_array[:split_idx, :, :]  # Training input
test_xs = x_array[split_idx:, :, :]   # Testing input
train_ys = y_array[:split_idx, :, :]  # Training labels
test_ys = y_array[split_idx:, :, :]   # Testing labels


train_xs = np.swapaxes(train_xs, 0, 1)
test_xs = np.swapaxes(test_xs, 0, 1)
train_ys = np.swapaxes(train_ys, 0, 1)
test_ys = np.swapaxes(test_ys, 0, 1)
print(f'train_xs.shape = {train_xs.shape}')
print(f'train_xs = {train_xs[:,0]}')
print(f'train_ys.shape = {train_ys.shape}')
print(f'train_ys = {train_ys[:,0]}')
print(f'test_xs.shape = {test_xs.shape}')


#batch_size = 1
df_human = df
# Instantiate dataset
dataset_train_hu = DatasetRNN(train_xs, train_ys)
dataset_test_hu = DatasetRNN(test_xs, test_ys)
