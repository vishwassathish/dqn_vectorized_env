import numpy as np
import gymnasium as gym

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
    
        obs, info = self.env.reset(**kwargs)

        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _, _ = self.env.step(self.noop_action)
            if done:
                obs, _ = self.env.reset(**kwargs)
        return obs, info

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            # print("Resetting environment ... ")
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        # print("Reset lives left: ", self.lives)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        # print("step : ", done, self.lives)
        return obs, reward, done, truncated, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset/loss-of-life for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        self.lives = self.env.unwrapped.ale.lives()
        # assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        # assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _, info = self.env.step(1)
        if done:
            print("oops done ... ")
            self.env.reset(**kwargs)
            obs, _, done, _, info = self.env.step(1)
        if done:
            print("oops done again ... ")
            obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

    def step(self, ac):
        lives = self.env.unwrapped.ale.lives()

        # Take a dummy action when life is lost
        if lives < self.lives and lives > 0:
            obs, _, done, _, info = self.env.step(1)
            self.lives = lives
            if done:
                self.reset()
        
        elif lives == 0:
            self.reset()

        return self.env.step(ac)