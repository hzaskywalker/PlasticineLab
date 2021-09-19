import numpy as np
import cv2 as cv

class Interface:
    def __init__(self,action_dim):
        self.action_dim = action_dim
        self.action = np.zeros(self.action_dim,dtype=np.float64)

    def __call__(self,key):
        raise NotImplementedError

class ChopsticksInterface(Interface):
    def __init__(self):
        super(ChopsticksInterface,self).__init__(7)
        self.cursor = 0
        self.cnt = 0

    def __call__(self):
        key = cv.waitKey(0)
        if key == ord('a'):
            self.action[0] -= 5
        elif key == ord('d'):
            self.action[0] += 5
        elif key == ord('w'):
            self.action[2] += 5
        elif key == ord('s'):
            self.action[2] -= 5
        elif key == ord('q'):
            self.action[4] -= 5
        elif key == ord('e'):
            self.action[4] += 5
        elif key == ord('i'):
            self.action[1] += 5
        elif key == ord('k'):
            self.action[1] -= 5
        elif key == ord(' '):
            self.action = np.zeros(self.action_dim,dtype=np.float64)
        elif key == ord('x'):
            return None
        self.action.clip(-10,10)
        return self.action

    def sin(self):
        cv.waitKey(1)
        self.action[4] = 10*np.sin(self.cursor)
        self.cursor += np.pi/16
        return self.action

    def shake(self):
        cv.waitKey(1)
        self.action[1] = 2 if self.cnt < 5 else 10*np.sin(self.cursor)
        self.cursor += 0 if self.cnt < 5 else np.pi/16
        self.cnt += 1
        return self.action

    def swing(self):
        cv.waitKey(1)
        self.action[0] = 10*np.sin(self.cursor)
        self.cursor += np.pi/16
        return self.action

    def stab(self):
        cv.waitKey(1)
        self.action[2] = 10*np.sin(self.cursor)
        self.cursor += np.pi/16
        return self.action

    def squeeze(self):
        cv.waitKey(1)
        self.action[5] = -5*np.sin(self.cursor)
        self.cursor += np.pi/16
        return self.action