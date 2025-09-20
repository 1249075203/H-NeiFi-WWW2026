class Args:
    # Epoch
    training_episodes = 200
    # test_num = 10

    # Max step
    max_step = 35

    # num of non expert
    agent_num = 40

    # unuse
    batch_size = agent_num  # batch_size
    buffer_size = agent_num

    # Init. opinion of experts
    agent_exp = [1,3]

    learning_rate = 0.0001
    gamma = 0.9

    # X_min
    opinions_begin = 0
    # X_max
    opinions_end = 4

    #Global goal
    target_opinions = 1.5

    # q
    weight_normal = 0.9
    # p
    weight_exp = 0.1


    input_dim = 4
    h_dim = 36
    emb_dim = 16

    # bounded confidence
    r_c= 1
    # Merging threshold
    sg_c = 0.5


    # noisy = 0.2