# H-NeiFi

+ ```
  arguments.py : Parameter congfiguration files
  ```

  Regarding the number of agents, the range of opinions, user stubbornness, and other experimental configurations are provided in the experimental section of the text.

  When `weight_normal = 1, weight_exp = 0`, and `agent_confident = 0` (run_all_fast.py), the model degenerates into the HK model.
  
+ ```
  run_all.py: H-NeiFi
  ```

  

+ ```
  run_all_fast.py: H-NeiFi with GPU
  ```

  **Main executable file**

+ ```
  algorithm.py: policy graident
  ```

  

+ ```
  network.py: Bi-LSTM network
  ```

  

+ ```
  reward_funciton.py: reward setting
  ```

  