import pandas as pd
import numpy as np
from scipy.optimize import minimize
import bandits
import disrnn
import optax
from rnn_utils import DatasetRNN, compute_log_likelihood, train_model
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import matplotlib.pyplot as plt


##########################################
# SETTINGS: Choose optimization mode
##########################################
# Set to True to optimize parameters per participant,
# or False to optimize once for the entire dataset.
def run_optimization():

    OPTIMIZE_PER_PARTICIPANT = False

    ##########################################
    # LOAD DATA
    ##########################################
    df = pd.read_csv('/Users/marvinmathony/PycharmProjects/IntrinsicMot/data/human_data.csv')
    df['state'] = df.index % 8800  # Ensure state values are within range
    df['choice'] = df['choice'].astype(int) - 1
    #df["choice"] = 1 - (df["choice"] - 1)

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
            choice = int(row['choice'])  # Convert human choice from 1,2 to 0,1
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
                choice = int(row['choice'])
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
            choice = int(row['choice'])
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
        """simulated_datasets_train['global'] = dataset_train
        simulated_experiments_train['global'] = experiment_list_train
        simulated_datasets_test['global'] = dataset_test
        simulated_experiments_test['global'] = experiment_list_test"""

    ##########################################
    # CONCATENATE SIMULATED DATASETS (Row-wise)
    ##########################################
    # Assume that each DatasetRNN object holds its data in ._xs and ._ys arrays.
    # We will concatenate the underlying arrays along the episode dimension.
    """    all_xs = np.concatenate(
            [dataset._xs for dataset in simulated_datasets_train.values()], axis=1
        )
        all_ys = np.concatenate(
            [dataset._ys for dataset in simulated_datasets_train.values()], axis=1
        )
        print("Shape of concatenated xs:", all_xs.shape)
        print("Shape of concatenated ys:", all_ys.shape)

        # Create one large DatasetRNN object containing data for all participants.
        big_dataset = DatasetRNN(all_xs, all_ys)
        print("Combined DatasetRNN created with", big_dataset._dataset_size, "episodes.")"""

    ##########################################
    # CREATE DATAFRAME FROM SIMULATED EXPERIMENTS
    ##########################################
    rows = []
    for session in experiment_list_train:
        for i in range(session.n_trials):  # Loop through trials in each session
            rows.append({
                "Action": session.choices[i],
                "rewards": session.rewards[i],
                "timeseries": session.timeseries[i],
                "n_trials": session.n_trials,
                "V_t": session.V_t[i],
                "TU": session.TU[i],
                "RU": session.RU[i]
            })

    df_bandits = pd.DataFrame(rows)


    ##### PREPROCESSING HUMAN DATA #####

    #@title Load and Preprocess Human Data for RNN training

    # Load data
    df = pd.read_csv('/Users/marvinmathony/PycharmProjects/IntrinsicMot/data/human_data.csv')

    human = True

    if human:
        n_participants = 44
        df.columns = ['Participant', 'Block', 'Trial', 'mu1', 'mu2', 'Action', 'Reward', 'RT']
        #df["Action"] = 1 - (df["Action"] - 1) # to try because of probit
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

    return dataset_train, dataset_train_hu, dataset_test_hu, df_human, n_trials_per_session, n_sessions, df_bandits


######### set up the model #########

#@title Set up Disentangled RNN.
#@markdown Number of latent units in the model.
latent_size = 5  #@param

#@markdown Number of hidden units in each of the two layers of the update MLP.
update_mlp_shape = (3,3,)  #@param

#@markdown Number of hidden units in each of the two layers of the choice MLP.
choice_mlp_shape = (2,)



def make_disrnn():
    model = disrnn.HkDisRNN(latent_size = latent_size,
                        update_mlp_shape = update_mlp_shape,
                        choice_mlp_shape = choice_mlp_shape,
                        target_size=2)
    return model

def make_disrnn_eval():
    model = disrnn.HkDisRNN(latent_size = latent_size,
                        update_mlp_shape = update_mlp_shape,
                        choice_mlp_shape = choice_mlp_shape,
                        target_size=2,
                        eval_mode=True)
    return model


optimizer = optax.adam(learning_rate=1e-2)
n_steps = 3000  #@param
information_penalty = 1e-03  #@param

def fit_disentangled(dataset):

    disrnn_params, opt_state, losses = train_model(
        model_fun = make_disrnn,
        dataset = dataset,
        optimizer = optimizer,
        loss_fun = 'penalized_categorical',
        penalty_scale=information_penalty,
        n_steps=n_steps,
    )
    return disrnn_params, opt_state, losses

def forward_simulate_network(network_params, n_trials_per_session, n_sessions):
    _, experiment_list_train, kalman_list_gru = bandits.create_dataset(
        agent_cls=bandits.AgentNetwork(make_disrnn, network_params),  # Pass instance directly
        env_cls=bandits.GershmanBandit,
        n_trials_per_session=n_trials_per_session,
        n_sessions=n_sessions,
        env_kwargs=env_params)
    return experiment_list_train, kalman_list_gru

def create_df(kalman_list, experiment_list):

    trained_network_probit = []
    for i in range(len(kalman_list)):
        for j in range(len(kalman_list[i][1])):
            trained_network_probit.append({
                'V_t': kalman_list[i][2][j],
                'Action': experiment_list[i][0][j],
                'RU': kalman_list[i][1][j,0] - kalman_list[i][1][j,1],
                'TU': kalman_list[i][1][j,0] + kalman_list[i][1][j,1],
                'posterior_std_0': kalman_list[i][1][j,0]})

    df = pd.DataFrame(trained_network_probit)
    return df

def run_probit_regression(df, bandit_network):
    # Flip the action variable if needed
    # TODO: implement conditional, depending on structure of dataset
    df = df.copy()

    #df['Action'] = 1 - df['Action']

    # Prepare the data
    X = df[['V_t', 'RU']].copy()
    X['V/TU'] = X['V_t'] / df['TU']
    y = df['Action']  # Binary outcome variable

    # Fit a probit regression using logistic regression on a normal CDF
    model = LogisticRegression()
    model.fit(X, 1- y) 
    # Extract coefficients
    w1, w2, w3 = model.coef_[0]
    print(f'Coefficients: w1={w1}, w2={w2}, w3={w3}')

    # Generate predictions
    df['Predicted_Prob'] = norm.cdf(w1 * X['V_t'] + w2 * X['RU'] + w3 * X['V/TU'])

    # Return updated DataFrame and coefficients
    return df, (w1, w2, w3)

### global parameters ###
env_params = {
        'innov_variance': 100,
        'noise_variance': 10,
        'n_actions': 2
    }

def plot_probit_regression(w1, w2, w3, df, title):
    # Fix RU and TU to their mean values
    RU_fixed = df['RU'].mean()
    TU_fixed = df['TU'].mean()

    # Generate a range of V_t values
    V_range = np.linspace(-100, 100, 1000)

    # Predict probabilities for each V_t
    predicted_probs = norm.cdf(
        w1 * V_range + w2 * RU_fixed + w3 * (V_range / TU_fixed)
    )

    # Plot the probit regression curve
    plt.figure(figsize=(8, 6))
    plt.plot(V_range, predicted_probs, label="Probit Regression Curve", color='black')
    plt.xlabel('Expected Value Difference (V)')
    plt.ylabel('Choice Probability')
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_probit_regression_median(w1, w2, w3, df, title, UCB):
  if UCB:
    U_median = df['RU'].median()
    U = df['RU']
  else:
    U_median = df['TU'].median()
    U = df['TU']

  # Split the data into low and high SD groups
  low_SD = df[U <= U_median]
  high_SD = df[U > U_median]

  # Generate a range of V_t values
  V_range = np.linspace(-100, 100, 1000)

  RU_low_SD = low_SD['RU'].mean()
  RU_high_SD = high_SD['RU'].mean()
  TU_low_SD = low_SD['TU'].mean()
  TU_high_SD = high_SD['TU'].mean()

  # Predict probabilities for low SD group
  predicted_probs_low_SD = norm.cdf(
      w1 * V_range + w2 * RU_low_SD + w3 * (V_range / TU_low_SD)
  )

  # Predict probabilities for high SD group
  predicted_probs_high_SD = norm.cdf(
      w1 * V_range + w2 * RU_high_SD + w3 * (V_range / TU_high_SD)
  )

  # Plot the probit regression curves for low and high SD
  plt.figure(figsize=(8, 6))
  plt.plot(V_range, predicted_probs_low_SD, label="Low SD", color='black', linewidth=2)
  plt.plot(V_range, predicted_probs_high_SD, label="High SD", color='gray', linewidth=2)
  plt.xlabel('Expected Value Difference (V)')
  plt.ylabel('Choice Probability')
  plt.title(title)
  plt.ylim(0, 1)
  plt.legend()
  plt.grid(True)
  plt.show()





if __name__ == '__main__':
    
    optimization = True

    if optimization:
        dataset_train, dataset_train_hu, dataset_test_hu, df_human, n_trials_per_session, n_sessions, df_bandits = run_optimization()

    # Number of repetitions
    n_repetitions = 10

    # Store results
    probit_coefficients = []
    normalized_likelihoods = []
    accuracies = []
    average_psuedo_r2 = []
    step = 0
    UCB = False

    for _ in range(n_repetitions):
        step += 1
        print(f'Step {step}')
        # Fit disentangled model
        disrnn_params_hu, _, _ = fit_disentangled(dataset_train_hu)
        disrnn_params_bandits, _, _ = fit_disentangled(dataset_train)

        # Compute log likelihoods
        likelihood_hu, _, _, _, accuracy_hu, pseudor2_hu = compute_log_likelihood(dataset_test_hu, make_disrnn_eval, disrnn_params_hu)
        likelihood_bandits, _, _, _, accuracy_bandits, pseudor2_bandits = compute_log_likelihood(dataset_test_hu, make_disrnn_eval, disrnn_params_bandits)

        normalized_likelihoods.append((likelihood_hu, likelihood_bandits))
        accuracies.append((accuracy_hu, accuracy_bandits))

        average_psuedo_r2.append((pseudor2_hu, pseudor2_bandits))


        # Forward simulation
        experiment_list_hu, kalman_list_hu = forward_simulate_network(disrnn_params_hu, n_trials_per_session, n_sessions)
        experiment_list_bandits, kalman_list_bandits = forward_simulate_network(disrnn_params_bandits, n_trials_per_session, n_sessions)

        # Create dataframes
        df_trained_agent_hu = create_df(kalman_list_hu, experiment_list_hu)
        df_trained_agent_bandits = create_df(kalman_list_bandits, experiment_list_bandits)

        # Run probit regression
        df_human, coefficients_hu = run_probit_regression(df_trained_agent_hu, False)
        df_bandits_network, coefficients_bandits = run_probit_regression(df_trained_agent_bandits, True)
        df_bandits_not_net, coefficients_bandits_not_net = run_probit_regression(df_bandits, False)

        w1_hu, w2_hu, w3_hu = coefficients_hu
        w1_bandits, w2_bandits, w3_bandits = coefficients_bandits
        w1_bandits_not_net, w2_bandits_not_net, w3_bandits_not_net = coefficients_bandits_not_net

        probit_coefficients.append((coefficients_hu, coefficients_bandits))

        # Plot probit regression
        # Plot for df_human - disRNN trained on human data
        plot_probit_regression(w1_hu, w2_hu, w3_hu, df_human, "Probit Regression: DisRNN trained on Human Data")

        # Plot for df_bandits -disRNN trained on hybrid bandit data
        plot_probit_regression(w1_bandits, w2_bandits, w3_bandits, df_bandits_network, "Probit Regression: DisRNN trained on Bandits Data")

        # Plot for bandits without NN
        plot_probit_regression(w1_bandits_not_net, w2_bandits_not_net, w3_bandits_not_net, df_bandits, "Probit Regression: Bandits Data (no NN)")

        # Plot for df_human
        plot_probit_regression_median(w1_hu, w2_hu, w3_hu, df_human, "Probit Regression Median Split: Human Data - DisRNN", UCB)

        # Plot for df_bandits
        plot_probit_regression_median(w1_bandits, w2_bandits, w3_bandits, df_bandits_network, "Probit Regression Median Split: Bandits Data - DisRNN", UCB)

        # Plot for bandits without NN
        plot_probit_regression_median(w1_bandits_not_net, w2_bandits_not_net, w3_bandits_not_net, df_bandits, "Probit Regression Median Split: Bandits Data (no NN)", UCB)

        #plot latent dynamics of RNN
        #plot bottlenecks for human trained NNs
        title_human = "Agent trained on Human Data"
        disrnn.plot_bottlenecks(disrnn_params_hu, title_human)
        #plt.savefig("bottlenecks_thompson.png", dpi=300)

        #plot bottlenecks for hybrid agent trained NNs
        title_bandits = "Agent trained on Bandits Data"
        disrnn.plot_bottlenecks(disrnn_params_bandits, title_bandits)
        #plt.savefig("bottlenecks_thompson.png", dpi=300)

        print('now plotting human data update rules')
        disrnn.plot_update_rules(disrnn_params_hu, make_disrnn_eval, title_human)
        print('now plotting human data update rules')

        #plt.savefig("update_rules_human.png", dpi=300)

        print('now plotting bandit data update rules')
        disrnn.plot_update_rules(disrnn_params_bandits, make_disrnn_eval, title_bandits)
        #plt.savefig("update_rules_bandits.png", dpi=300)

    # Compute averages
    probit_coefficients = np.array(probit_coefficients)
    avg_coefficients_hu = np.mean(probit_coefficients[:, 0], axis=0)
    avg_coefficients_bandits = np.mean(probit_coefficients[:, 1], axis=0)

    normalized_likelihoods = np.array(normalized_likelihoods)
    avg_likelihood_hu = np.mean(normalized_likelihoods[:, 0])
    avg_likelihood_bandits = np.mean(normalized_likelihoods[:, 1])

    average_psuedo_r2 = np.array(average_psuedo_r2)
    avg_pseudo_r2_hu = np.mean(average_psuedo_r2[:, 0])
    avg_pseudo_r2_bandits = np.mean(average_psuedo_r2[:, 1])

    accuracies = np.array(accuracies)
    avg_accuracy_hu = np.mean(accuracies[:, 0])
    avg_accuracy_bandits = np.mean(accuracies[:, 1])

    # compute se's
    n = n_repetitions

    se_coefficients_hu = np.std(probit_coefficients[:, 0], axis=0, ddof=1) / np.sqrt(n)
    se_coefficients_bandits = np.std(probit_coefficients[:, 1], axis=0, ddof=1) / np.sqrt(n)

    se_likelihood_hu = np.std(normalized_likelihoods[:, 0], ddof=1) / np.sqrt(n)
    se_likelihood_bandits = np.std(normalized_likelihoods[:, 1], ddof=1) / np.sqrt(n)

    se_pseudo_r2_hu = np.std(average_psuedo_r2[:, 0], ddof=1) / np.sqrt(n)
    se_pseudo_r2_bandits = np.std(average_psuedo_r2[:, 1], ddof=1) / np.sqrt(n)

    se_accuracy_hu = np.std(accuracies[:, 0], ddof=1) / np.sqrt(n)
    se_accuracy_bandits = np.std(accuracies[:, 1], ddof=1) / np.sqrt(n)

    print(f'Average Probit Coefficients (Human): {avg_coefficients_hu} ± {se_coefficients_hu}')
    print(f'Average Probit Coefficients (Bandits): {avg_coefficients_bandits} ± {se_coefficients_bandits}')
    print(f'Average Normalized Likelihood (Human): {avg_likelihood_hu} ± {se_likelihood_hu}')
    print(f'Average Normalized Likelihood (Bandits): {avg_likelihood_bandits} ± {se_likelihood_bandits}')
    print(f'Average Accuracy (Human): {avg_accuracy_hu} ± {se_accuracy_hu}')
    print(f'Average Accuracy (Bandits): {avg_accuracy_bandits} ± {se_accuracy_bandits}')
    print(f'Average Pseudo R2 (Human): {avg_pseudo_r2_hu} ± {se_pseudo_r2_hu}')
    print(f'Average Pseudo R2 (Bandits): {avg_pseudo_r2_bandits} ± {se_pseudo_r2_bandits}')
