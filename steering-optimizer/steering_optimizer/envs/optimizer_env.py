import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from sympy.core.symbol import symbols
from sympy.solvers import solve
from sympy import Symbol

class StrOptEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.WB = WB
    self.TW = TW
    self.KP = KP
    self.tr_min = tr_min

    self.WLX = -TW/2
    self.WLY = 0.0

    self.KPLX = WLX+KP
    self.KPLY = WLY

    self.A = 100
    self.A_ang = -75
    self.Dx = -100
    self.Dy = -200

    self.tr_eval = np.inf
    self.error = np.inf


    self.observation_space = [self.A, self.A_ang, self.Dx, self.Dy, self.WB, self.TW, self.KP, self.tr_eval, self.error]
    self.action_space = [0,'A+', 'A_ang+', 'Dx+', 'Dy+', 'A-', 'A_ang-', 'Dx-', 'Dy-']

    Ax, Ay, Tx, Ty, T, A, z, Dy = symbols('Ax, Ay, Tx, Ty, T, A, z, Dy')

    eq1 = self.KPLX + Ax + Tx - z
    eq2 = self.KPLY + Ay + Ty - Dy
    eq3 = Ax**2 + Ay**2 - A**2
    eq4 = Tx**2 + Ty**2 - T**2
    system = [eq4, eq2, eq3, eq1]

    solution = solve(system, [Ax, Ay, Tx, Ty, z])

    Ax_solutions = [sol[Ax] for sol in solution]
    Ay_solutions = [sol[Ay] for sol in solution]

  def step(self, action):

    #INPUT LIST

    # [0] ARM length, [1] ARM angle in degrees, [2] Rack endpoint distance from middle axis, [3] Rack distance from front axis

    input_list = [100, -45, -100, -100]

    #Rack parameters

    Dx = input_list[2]
    Dy_eval = input_list[3]


    #Arm length and angle
    A_eval = input_list[0]
    angle_deg = input_list[1]


    original_ARM_angle = angle_deg/180*np.pi

    #Arm coordinates
    Ax0 = KPLX + A_eval*np.cos(original_ARM_angle)
    Ay0 = KPLY + A_eval*np.sin(original_ARM_angle)

    #tierod length
    T_eval = np.sqrt(np.power((Ax0-Dx),2)+np.power((Ay0-Dy_eval),2))

    #MAXIMAL TRAVEL DISTANCE

    Dmax = T_eval + A_eval
    DmaxX_dist = np.sqrt(np.power(Dmax,2) - np.power(Dy_eval,2))
    DmaxX = DmaxX_dist + KPLX
    travelmax = DmaxX - Dx


    Dmin = T_eval - A_eval
    DminX_dist = np.sqrt(np.power(Dmin,2) - np.power(Dy_eval,2))
    DminX = DminX_dist + KPLX
    travelmin = DminX - Dx

    if np.abs(travelmax) > np.abs(travelmin):
        travel = np.abs(travelmin)
    else:
        travel = np.abs(travelmax)

    #Initial angle orientation calc

    szog = np.arctan((KPLY-Dy_eval)/(KPLX-Dx))/np.pi*180
    szog_diff = angle_deg - szog
    szog_diff_rad = szog_diff/180*np.pi

    #Substituting basic parameters
    import sympy

    Ax_solutions_simp = []
    Ay_solutions_simp = []


    Ax_solutions_simp.append(sympy.simplify(Ax_solutions[0]).subs({"T":T_eval, "Dy":Dy_eval, "A":A_eval})) 
    Ax_solutions_simp.append(sympy.simplify(Ax_solutions[1]).subs({"T":T_eval, "Dy":Dy_eval, "A":A_eval})) 

    Ay_solutions_simp.append(sympy.simplify(Ay_solutions[0]).subs({"T":T_eval, "Dy":Dy_eval, "A":A_eval}))
    Ay_solutions_simp.append(sympy.simplify(Ay_solutions[1]).subs({"T":T_eval, "Dy":Dy_eval, "A":A_eval}))

    while x < travel:
      z_eval = Dx + x
      z_array = np.append(z_array,z_eval-Dx)

      z_eval_c = Dx - x

      Ax_0 = float(eval_eqn(Ax_solutions_simp[0], {"z":z_eval}))
      Ay_0 = float(eval_eqn(Ay_solutions_simp[0], {"z":z_eval}))

      ARM_f1 = (np.arctan2(Ay_0,Ax_0)-original_ARM_angle)*180/np.pi
    
      if (szog_diff_rad > 0 and (ARM_f1-szog/180*np.pi) > 0) or (szog_diff_rad < 0 and (ARM_f1-szog/180*np.pi) < 0):

        Ax_0_c = float(eval_eqn(Ax_solutions_simp[0], {"z":z_eval_c}))
        Ay_0_c = float(eval_eqn(Ay_solutions_simp[0], {"z":z_eval_c}))

        ARM_f1c = -(np.arctan2(Ay_0_c,Ax_0_c)-original_ARM_angle)*180/np.pi

        #ARM_f1 = (f1-original_ARM_angle)*180/np.pi 
        #ARM_f1c = -(f1c-original_ARM_angle)*180/np.pi
        
      else:
        Ax_1 = float(eval_eqn(Ax_solutions_simp[1], {"z":z_eval}))
        Ay_1 = float(eval_eqn(Ay_solutions_simp[1], {"z":z_eval}))

        ARM_f1 = (np.arctan2(Ay_1,Ax_1)-original_ARM_angle)*180/np.pi

        Ax_1_c = float(eval_eqn(Ax_solutions_simp[1], {"z":z_eval_c}))
        Ay_1_c = float(eval_eqn(Ay_solutions_simp[1], {"z":z_eval_c}))
      
        ARM_f1c = -(np.arctan2(Ay_1_c,Ax_1_c)-original_ARM_angle)*180/np.pi

        #ARM_f1 = (f2-original_ARM_angle)*180/np.pi
        #ARM_f1c = -(f2c-original_ARM_angle)*180/np.pi
    
      if ARM_f1 > 180:
        ARM_f1 = ARM_f1 - 360
    
      if ARM_f1c > 180:
        ARM_f1c = ARM_f1c - 360
    
      l_array = np.append(l_array,ARM_f1)
      r_array = np.append(r_array,ARM_f1c)
    
      x = x+travel/25
    
    solv = z_array

    # Inner tire angle when turning left
    l = l_array

    # Outer tire angle when turning left
    r = r_array

    k = []

    for i in range(0,len(solv),1):
      k.append(np.arctan(WB/(WB/(np.tan(r[i]/180*np.pi))-TW))/np.pi*180)
    
    self.tr_eval = WB/np.arcsin(max(r)/180*np.pi)

    state = [self.A, self.A_ang, self.Dx, self.Dy, self.WB, self.TW, self.KP, self.tr_eval, self.error]

    return np.array(state, dtype=np.float32), reward, done, {}

  def reset(self):

    x = 0
    z_array = np.array([])
    l_array = np.array([])
    r_array = np.array([])

    return self.step(0)[0]

  def render(self, mode='human', close=False):
    return
  def eval_eqn(eqn,in_dict):
    subs = {sympy.symbols(key):item for key,item in in_dict.items()}
    ans = sympy.simplify(eqn).evalf(subs = subs)

    return ans
    