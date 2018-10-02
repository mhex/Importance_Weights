'''
SMO for importance weights.
'''
import numpy as np
from numba import jit
import sys


# param dE: Matrix of Gradients of the loss function with respect to parameters
#          dim(dE) = [ num_weights, num_samples  ]
# param p: Free parameter > 0
# param E: E absolut erreors of missclassification
# param C: upper bound for alpha

def compare_alphas(a1,a2, eps):
    arr = np.abs(a1-a2) < eps
    #ret = arr.all()
    #print "DEBUG:::ret = " + str(ret)
    return arr
#-------------------------------------------------------------------------------

#@jit
def smo(dE,p,E,C,max_iter=np.inf):
    #init alphas:
    dim_dE = dE.shape
    n_samples = dim_dE[1] # for convenience
    alpha = np.zeros((dim_dE[1],))
    old_alpha = np.zeros((dim_dE[1],))

    # calc kernel matrix for debugging:
    #K_DEBUG = dE.transpose().dot(dE)

    # init kernel matrix
    K = np.eye(n_samples,n_samples)

    # init addiditonal variables
    ubound = np.ones((n_samples,))*C
    lbound = np.zeros((n_samples,))
    notset = [True]*n_samples
    eps_zero = 1e-7
    eps = 1.0-eps_zero      # for comparison: x >= 1 --> x > eps
    roua1 = 0 # residuals
    roua2 = 0 #
    alpha_stat = np.zeros((dim_dE[1],)) # tracks change of alpha

    # scale problem such that K_ii = 1
    scaler = np.zeros((n_samples,))
    l = np.zeros((n_samples,))
    s = 0
    for i in range(n_samples):
        s = np.sqrt(dE[:,i].dot(dE[:,i]))
        l[i] = p*E[i]/s # since alpha is multiplied from left and right to K
        ubound[i] *= s       # scale upper constraint for alpha_i
        scaler[i] = s

    #print "DBEUG:::sum(ubound) = " + str(sum(ubound))


    F = np.array(-l)     # since alphas are all 0

    min_improv = 1e-3

    # calculate alphas
    EE=0  # objective is zero
    tol=0.1
    tolab=0.001       # minimalobjective improvements
    maxup=5.0*tolab
    avup = 5.0*tolab  # improvement of objective
    stepp=0
    EEo=EE


    ind_alpha1 = 0
    ind_alpha2 = 0
    steps = 0
    max_alphas_not_changed = 5
    n_alphas_not_changed = 0
    bad_cond = False

    debug_eig = []

    while (avup > tolab) and (steps < max_iter):
    #while (avup > tolab) and ( n_alphas_not_changed < max_alphas_not_changed):

        old_alpha = alpha.copy()

        # choose index of sample for alpha1 -> kkk-conditions
        maxup=-1.0
        for i in range(n_samples):
            tmp_f = F[i]
            xi = ubound[i]-alpha[i]
            if (tmp_f < 0.0) and (alpha[i] < (ubound[i]-eps_zero)): # check for  alpha = C => F <= 0
                tmp_f*=-1
                if tmp_f > xi:
                    tmp_f = xi
                if tmp_f > maxup:        # set
                    maxup = tmp_f
                    ind_alpha1 = i
            else:
                xi = alpha[i]-lbound[i]
                if (tmp_f > 0.0) and (alpha[i] > (lbound[i]+eps_zero)):
                    if (tmp_f > xi):
                        tmp_f = xi
                    if tmp_f > maxup:
                        maxup=tmp_f
                        ind_alpha1 = i

        #print "DEBUG:::ind_alpha1 = " + str(ind_alpha1)

        f1 = F[ind_alpha1]
        a1 = alpha[ind_alpha1]
        lb1 = lbound[ind_alpha1]
        ub1 = ubound[ind_alpha1]

        # calculate kernel entry for ind_alpha1
        if notset[ind_alpha1]:
            notset[ind_alpha1] = False
            for i in range(n_samples):
                if notset[i]:   # calculate only examples not already calculated
                    kij = dE[:,ind_alpha1].dot(dE[:,i])
                    kij /= scaler[ind_alpha1]*scaler[i]
                    K[ind_alpha1,i] = kij
                    K[i,ind_alpha1] = kij


        # search for index alpha2
        #maxup = -1.0
        maxup = 0.0
        for i in range(n_samples):
            k12 = K[ind_alpha1,i] # 0 <= k12 <= 1 because we scaled the matrix!
            if k12 > eps:
                k12 = eps
            tmp = 1.0/(1.0-k12*k12)
            f2 = F[i]
            opt_unbound_a1 = (f2*k12-f1)*tmp
            opt_unbound_a2 = (f1*k12-f2)*tmp

            #if steps == 0:
            #    print "DEBUG:::i = " + str(i)
            #    print "DEBUG:::opt_unbound_a1 + a1 = " + str((opt_unbound_a1 + a1)/scaler[i])
            #    debug_eig.append((opt_unbound_a1 + a1)/scaler[i])

            # check if updates are on bound
            # -1 -> not on bound
            #  0 -> on lower bound
            #  1 -> on upper bound
            b1 = -1  # bound information for alpha1
            b2 = -1  # bound indformation for alpha2

            # binary bound indicator: 0 -> not on bound, 1 -> on bound
            onbound1 = 0 # bound indicator of alpha1
            onbound2 = 0 # bound indicator of alpha2


            # violations of constraints for updates are checked
            # and corrected for alpha1 and alpha2
            if (opt_unbound_a1 + a1) < lb1:
                opt_unbound_a1 = lb1 - a1
                onbound1 = 1
                b1 = 0
            elif (opt_unbound_a1 + a1) > ub1:
                opt_unbound_a1 = ub1 - a1
                onbound1 = 1
                b1 = 1

            if (opt_unbound_a2 + alpha[i]) < lbound[i]:
                opt_unbound_a2 = lbound[i] - alpha[i]
                onbound2 = 1
                b2 = 0
            elif (opt_unbound_a2 + alpha[i]) > ubound[i]:
                opt_unbound_a2 = ubound[i] - alpha[i]
                onbound2 = 1
                b2 = 1

            # bidk (bound indikator) indikates if:
            # no alpha is on bound     = 0
            # only alpha1 is on bound  = 1
            # only alpha2 is on bound  = 2
            # both alphas are on bound = 3
            bidk = 2*onbound2+onbound1
            #print "DEBUG:::bound_indikator = " + str(bidk)
            if bidk > 0:
                if bidk == 1:
                    opt_unbound_a2 = -(k12*opt_unbound_a1+f2)
                    if (opt_unbound_a2 + alpha[i]) < lbound[i]:
                        opt_unbound_a2 = lbound[i] - alpha[i]
                        onbound2 = 1
                        bidk = 3
                    elif (opt_unbound_a2 + alpha[i]) > ubound[i]:
                        opt_unbound_a2 = ubound[i] - alpha[i]
                        onbound2 = 1
                        bidk = 3
                elif bidk == 2:
                    opt_unbound_a1 = -(k12*opt_unbound_a2+f1)
                    if (opt_unbound_a2 + a1) < lb1:
                        opt_unbound_a1 = lb1 - a1
                        onbound1 = 1
                        bidk=3
                    elif (opt_unbound_a1 + a1) > ub1:
                        opt_unbound_a1 = ub1-a1
                        onbound1 = 1
                        bidk = 3
            if bidk == 3:
                # corner indicates, if:
                # on lower/lower corner  => 0
                # on upper/lower corner  => 1
                # on lower/upper corner  => 2
                # on upper/upper corner  => 3
                corner = 2*b2+b1
                if corner == 0:
                    g2 = opt_unbound_a2+opt_unbound_a1*k12+f2
                    if g2 < 0:
                        opt_unbound_a2 = - (k12*opt_unbound_a1+f2)
                        if (opt_unbound_a2 + alpha[i]) < lbound[i]:
                            opt_unbound_a2 = lbound[i] - alpha[i]
                        elif (opt_unbound_a2 + alpha[i]) > ubound[i]:
                            opt_unbound_a2 = ubound[i] - alpha[i]
                    else:
                        g1 = opt_unbound_a1+opt_unbound_a2*k12+f1
                        if g1 < 0:
                            opt_unbound_a1 = -(k12*opt_unbound_a2+f1)
                            if (opt_unbound_a1+a1) < lb1:
                                opt_unbound_a1 = lb1-a1
                            elif (opt_unbound_a1+a1) > ub1:
                                opt_unbound_a1 = ub1-a1
                if corner == 1:
                    g2 = opt_unbound_a2+opt_unbound_a1*k12+f2
                    if g2 < 0:
                        opt_unbound_a2 = -(k12*opt_unbound_a1+f2)
                        if (opt_unbound_a2+alpha[i]) < lbound[i]:
                            opt_unbound_a2 = lbound[i] - alpha[i]
                        elif(opt_unbound_a2 + alpha[i]) > ubound[i]:
                            opt_unbound_a2 = ubound[i]-alpha[i]
                    else:
                        g1 = opt_unbound_a1+opt_unbound_a2*k12+f1
                        if g1 > 0:
                            opt_unbound_a1 = -(k12*opt_unbound_a2+f1)
                            if (opt_unbound_a1+a1) < lb1:
                                opt_unbound_a1 = lb1-a1
                            elif (opt_unbound_a1+a1) > ub1:
                                opt_unbound_a1 = ub1-a1
                if corner == 2:
                    g2 = opt_unbound_a2+ opt_unbound_a1*k12+f2
                    if g2 > 0:
                        opt_unbound_a2 = -(k12*opt_unbound_a1 + f2)
                        if (opt_unbound_a2 + alpha[i]) < lbound[i]:
                            opt_unbound_a2 = lbound[i] - alpha[i]
                        elif (opt_unbound_a2+alpha[i] > ubound[i]):
                            opt_unbound_a2 = ubound[i] - alpha[i]
                    else:
                        g1= opt_unbound_a1+opt_unbound_a2*k12+f1
                        if g1 < 0:
                            opt_unbound_a1 = -(k12*opt_unbound_a2+f1)
                            if (opt_unbound_a1+a1) < lb1:
                                opt_unbound_a1 = lb1 -a1
                            elif (opt_unbound_a1+a1) > ub1:
                                opt_unbound_a1 = ub1 - a1
                if corner == 3:
                    g2 = opt_unbound_a2+opt_unbound_a1*k12+f2
                    if g2 > 0:
                        opt_unbound_a2 = -(k12*opt_unbound_a1 + f2)
                        if (opt_unbound_a2+ alpha[i]) < lbound[i]:
                            opt_unbound_a2 = lbound[i] - alpha[i]
                        elif (opt_unbound_a2 + alpha[i]) > ubound[i]:
                            opt_unbound_a2 = ubound[i] - alpha[i]
                    else:
                        g1 = opt_unbound_a1 + opt_unbound_a2*k12+f1
                        if g1 > 0:
                            opt_unbound_a1 = -(k12*opt_unbound_a2+f1)
                            if (opt_unbound_a1 + a1) < lb1:
                                opt_unbound_a1 = lb1 -a1
                            elif (opt_unbound_a1 + a1) > ub1:
                                opt_unbound_a1 = ub1 - a1


            mxa = abs(opt_unbound_a1)
            if mxa <  abs(opt_unbound_a2):
                mxa = abs(opt_unbound_a2)

            if mxa < eps_zero:
                roua1 = 0#opt_unbound_a1
                roua2 = 0#opt_unbound_a2
            elif mxa > maxup:
                #print "DEBUG:::mxa > maxup @ i = " + str(i)
                maxup = mxa
                roua1 = opt_unbound_a1
                roua2 = opt_unbound_a2
                ind_alpha2 = i

            # calculate kernel entry for ind_alpha2
            if notset[ind_alpha2]:
                notset[ind_alpha2] = False
                for j in range(n_samples):
                    if notset[j]:   # calculate only examples not already calculated
                        kij = dE[:,ind_alpha2].dot(dE[:,j])
                        kij /= (scaler[ind_alpha2]*scaler[j])
                        K[ind_alpha2,j] = kij
                        K[j,ind_alpha2] = kij


        k12 = K[ind_alpha1][ind_alpha2]
        if k12 > eps:
            k12 = eps
        f2 = F[ind_alpha2]
        a2 = alpha[ind_alpha2]
        lb2 = lbound[ind_alpha2]
        ub2 = ubound[ind_alpha2]

        if (roua1 + a1) > ub1:
            roua1 = ub1 -a1

        if (roua1 + a1) < lb1:
            roua1 = lb1 - a1

        if (roua2 + a2) > ub2:
            roua2 = ub2 - a2

        if (roua2 + a2) < lb2:
            roua2 = lb2-a2

        alpha[ind_alpha1] += roua1
        alpha[ind_alpha2] += roua2


        # update objective
        EE += roua1*(0.5*roua1 + f1 + roua2*k12) + roua2*(0.5*roua2 + f2)
        #sys.exit(0)
        #print "DEBUG:::newObjective = " + str(EE)

        # updates F
        F += roua1*K[:,ind_alpha1] + roua2*K[:,ind_alpha2]



        steps += 1
        #print("DEBBUG:::steps = " + str(steps))
        improvement = abs(EE-EEo) / (abs(EEo) + tolab)
        EEo=EE
        avup = 0.8*avup+0.2*improvement

        comp = compare_alphas(old_alpha,alpha,eps_zero)
        if comp.all():
            n_alphas_not_changed += 1
        else:
            n_alphas_not_changed = 0
            changed = np.where(comp==False)
            alpha_stat[changed] += 1

    #alpha = alpha / scaler
    alpha = old_alpha / scaler
    np.savetxt('K_smo.txt',K)
    np.savetxt('F.txt', F)
    return {'alphas': alpha, 'steps': steps, 'EE': EE, 'scaler': scaler, 'F':F, 'K':K, 'deb':debug_eig}
