from math import sqrt, atan, degrees, cos, sin, radians
from itertools import permutations
l = 533.4
L = 304.8
sb = 265.7
sp = 127
wb = (sqrt(3)/6)*sb
ub = (sqrt(3)/3)*sb
wp = (sqrt(3)/6)*sp
up = (sqrt(3)/3)*sp

def atand(x, y):
    return degrees(atan(x/ y))

def cosd(x):
    return cos(radians(x))

def sind(x):
    return sin(radians(x))

def threeSpheres(tup):
    a1v, a2v, a3v, r1, r2, r3 = tup
    valid = True
    if a3v[2] == a2v[2] and a3v[2] == a1v[2]:
        x1, y1, z1 = a1v
        x2, y2, z2 = a2v
        x3, y3, z3 = a3v
        a = 2*(x3 - x1)
        b = 2*(y3 - y1)
        c = r1**2 - r3**2 - x1**2 - y1**2 + x3**2 + y3**2
        d = 2*(x3 - x2)
        e = 2*(y3 - y2)
        f = r2**2 - r3**2 - x2**2 - y2**2 + x3**2 + y3**2

        xSoln = (c*e - b*f)/(a*e - b*d)
        ySoln = (a*f - c*d)/(a*e - b*d)

        B = -2*z1
        C = z1**2 - r1**2 + (xSoln-x1)**2 + (ySoln-y1)**2

        zPlusSoln = (-B + sqrt(B**2 - 4*C))/2
        zMinusSoln = (-B - sqrt(B**2 - 4*C))/2

        if B**2 - 4*C < 0:
            valid = False

        plusSoln = (xSoln, ySoln, zPlusSoln)
        minusSoln = (xSoln, ySoln, zMinusSoln)
        return plusSoln, minusSoln, valid

    elif a3v[2] == a2v[2]:
        a3v, a1v = a1v, a3v

    elif a3v[2] == a1v[2]:
        a3v, a2v = a2v, a3v

    x1, y1, z1 = a1v
    x2, y2, z2 = a2v
    x3, y3, z3 = a3v
    a11 = 2*(x3 - x1)
    a12 = 2*(y3 - y1)
    a13 = 2*(z3 - z1)
    a21 = 2*(x3 - x2)
    a22 = 2*(y3 - y2)
    a23 = 2*(z3 - z2)

    b1 = r1**2 - r3**2 -x1**2 - y1**2 - z1**2 + x3**2 + y3**2 + z3**2
    b2 = r2**2 - r3**2 -x2**2 - y2**2 - z2**2 + x3**2 + y3**2 + z3**2

    a1 = (a11/a13) - (a21/a23)
    a2 = (a12/a13) - (a22/a23)
    a3 = (b2/a23) - (b1/a13)

    a4 = -a2/a1
    a5 = -a3/a1

    a6 = (-a21*a4 - a22)/a23
    a7 = (b2 - a21*a5)/a23

    a = a4**2 + 1 + a6**2
    b = 2*a4*(a5 - x1) - 2*y1 + 2*a6*(a7 - z1)
    c = a5*(a5 - 2*x1) + a7*(a7 - 2*z1) + x1**2 + y1**2 - r1**2

    yPlusSoln = (-b+sqrt(b**2 - 4*a*c))/(2*a)
    yMinusSoln = (-b-sqrt(b**2 - 4*a*c))/(2*a)

    if b**2 - 4*a*c < 0:
        valid = False
    xPlusSoln = a4*yPlusSoln + a5
    xMinusSoln = a4*yMinusSoln + a5

    zPlusSoln = a6*yPlusSoln + a7
    zMinusSoln = a6*yMinusSoln + a7

    plusSoln = (xPlusSoln, yPlusSoln, zPlusSoln)
    minusSoln = (xMinusSoln, yMinusSoln, zMinusSoln)
    return plusSoln, minusSoln, valid

def getSphereCenters(t1, t2, t3):
    a1v = (0, -wb-L*cosd(t1)+up, -L*sind(t1))
    a2v = ((sqrt(3)/2)*(wb + L*cosd(t2))-sp/2, (1/2)*(wb + L*cosd(t2))-wp, -L*sind(t2))
    a3v = (-(sqrt(3)/2)*(wb + L*cosd(t3))+sp/2, (1/2)*(wb + L*cosd(t3))-wp, -L*sind(t3))
    return a1v, a2v, a3v, l, l, l

def fk(t1, t2, t3):
    plusSol, minusSol, valid = threeSpheres(getSphereCenters(t1, t2, t3))
    if minusSol[2] <= 0: return minusSol, valid
    elif plusSol[2] <= 0: return plusSol, valid


def decider(theta1_plus,theta1_minus,theta2_plus,theta2_minus,theta3_plus,theta3_minus):
    valid = False
    theta1s = (theta1_plus, theta1_minus)
    theta2s = (theta2_plus, theta2_minus)
    theta3s = (theta3_plus, theta3_minus)
    for t1 in theta1s:
        for t2 in theta2s:
            for t3 in theta3s:
                # print(t1, t2, t3)
                solution, isValid = fk(t1, t2, t3)
                # print(solution, isValid)
                if isValid and -90 < t1 < 90 and -90 < t2 < 90 and -90 < t3 < 90: 
                    return t1, t2, t3
    return None



def getAngles(x, y, z):
    a = wb-up
    b = (sp/2) - (sqrt(3)/2)*wb
    c = wp - (1/2)*wb
    E1 = 2*L*(y+a)
    F1 = 2*z*L
    G1 = x**2 + y**2 + z**2 + a**2 + L**2 + 2*y*a - l**2

    E2 = -L*((sqrt(3)*(x+b))+y+c)
    F2 = 2*z*L
    G2 = x**2 + y**2 + z**2 + b**2 + c**2 + L**2 + 2*(x*b+y*c) - l**2

    E3 = L*((sqrt(3)*(x-b))-y-c)
    F3 = 2*z*L
    G3 = x**2 + y**2 + z**2 + b**2 + c**2 + L**2 + 2*(-x*b+y*c) - l**2

    if E1**2 + F1**2 - G1**2 < 0 or E2**2 + F2**2 - G2**2  < 0 or E3**2 + F3**2 - G3**2 < 0:
        return None

    theta1_plus = 2*atand((-F1+sqrt(E1**2 + F1**2 - G1**2)), (G1-E1))
    theta1_minus = 2*atand((-F1-sqrt(E1**2 + F1**2 - G1**2)), (G1-E1))

    theta2_plus = 2*atand((-F2+sqrt(E2**2 + F2**2 - G2**2)),(G2-E2))
    theta2_minus = 2*atand((-F2-sqrt(E2**2 + F2**2 - G2**2)), (G2-E2))
    
    theta3_plus = 2*atand((-F3+sqrt(E3**2 + F3**2 - G3**2)), (G3-E3))
    theta3_minus = 2*atand((-F3-sqrt(E3**2 + F3**2 - G3**2)), (G3-E3))

    return decider(theta1_plus,theta1_minus,theta2_plus,theta2_minus,theta3_plus,theta3_minus)

def main():
    print(getAngles(0, 0, 0))

if __name__ == "__main__":
    main()
