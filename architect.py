import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

# self.optimizer is for architecture. optimizer is for weights


class Architect(object):

    def __init__(self, model_1, model_2, c_lambda, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model_1 = model_1
        self.model_2 = model_2

        self.c_lambda = c_lambda

        self.optimizer_1 = torch.optim.Adam(self.model_1.arch_parameters(),
                                            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        self.optimizer_2 = torch.optim.Adam(self.model_2.arch_parameters(),
                                            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer_1, network_optimizer_2):
        loss = self._get_loss_val(self, input, target)

        theta_1 = _concat(self.model_1.parameters()).data
        theta_2 = _concat(self.model_2.parameters()).data

        try:
            moment_1 = _concat(network_optimizer_1.state[v]['momentum_buffer']
                               for v in self.model_1.parameters()).mul_(self.network_momentum)
            moment_2 = _concat(network_optimizer_2.state[v]['momentum_buffer']
                               for v in self.model_2.parameters()).mul_(self.network_momentum)
        except:
            moment_1 = torch.zeros_like(theta_1)
            moment_2 = torch.zeros_like(theta_2)
        dtheta_1 = _concat(torch.autograd.grad(
            loss, self.model_1.parameters())).data + self.network_weight_decay*theta_1
        dtheta_2 = _concat(torch.autograd.grad(
            loss, self.model_2.parameters())).data + self.network_weight_decay*theta_2
        unrolled_model_1 = self._construct_model_from_theta(
            theta_1.sub(eta, moment_1+dtheta_1))
        unrolled_model_2 = self._construct_model_from_theta(
            theta_2.sub(eta, moment_2+dtheta_2))

        return unrolled_model_1, unrolled_model_2

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer_1, network_optimizer_2, unrolled):
        self.optimizer_1.zero_grad()
        self.optimizer_2.zero_grad()
        if unrolled:
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid, eta, network_optimizer_1, network_optimizer_2)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer_1.step()
        self.optimizer_2.step()

    def _backward_step(self, input_valid, target_valid):
        loss_1, _ = self.model_1._loss(input_valid, target_valid)
        loss_2, _ = self.model_2._loss(input_valid, target_valid)

        loss_1.backward()
        loss_2.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer_1, network_optimizer_2):
        unrolled_model_1, unrolled_model_2 = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer_1, network_optimizer_2)
        unrolled_loss_1, _ = unrolled_model_1._loss(input_valid, target_valid)
        unrolled_loss_2, _ = unrolled_model_2._loss(input_valid, target_valid)

        unrolled_loss = unrolled_loss_1 + unrolled_loss_2
        unrolled_loss.backward()

        dalpha_1 = [v.grad for v in unrolled_model_1.arch_parameters()]
        vector_1 = [v.grad.data for v in unrolled_model_1.parameters()]

        dalpha_2 = [v.grad for v in unrolled_model_2.arch_parameters()]
        vector_2 = [v.grad.data for v in unrolled_model_2.parameters()]

        implicit_grads_1 = self._hessian_vector_product(
            vector_1, vector_2, input_train, target_train, 1)
        implicit_grads_2 = self._hessian_vector_product(
            vector_1, vector_2, input_train, target_train, 2)

        for g, ig in zip(dalpha_1, implicit_grads_1):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model_1.arch_parameters(), dalpha_1):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        for g, ig in zip(dalpha_2, implicit_grads_2):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model_2.arch_parameters(), dalpha_2):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector_1, vector_2, input, target, index, r=1e-2):
        if index == 1:
            R = r / _concat(vector_1).norm()
            for p, v in zip(self.model_1.parameters(), vector_1):
                p.data.add_(R, v)
            loss = self._get_loss_val(self, input, target)
            grads_p = torch.autograd.grad(loss, self.model_1.arch_parameters())

            for p, v in zip(self.model_1.parameters(), vector_1):
                p.data.sub_(2*R, v)
            loss = self._get_loss_val(self, input, target)
            grads_n = torch.autograd.grad(loss, self.model_1.arch_parameters())

            for p, v in zip(self.model_1.parameters(), vector_1):
                p.data.add_(R, v)

            term_1 = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

            R = r / _concat(vector_2).norm()
            for p, v in zip(self.model_2.parameters(), vector_2):
                p.data.add_(R, v)
            loss = self._get_loss_val(self, input, target)
            grads_p = torch.autograd.grad(loss, self.model_1.arch_parameters())

            for p, v in zip(self.model_2.parameters(), vector_2):
                p.data.sub_(2*R, v)
            loss = self._get_loss_val(self, input, target)
            grads_n = torch.autograd.grad(loss, self.model_1.arch_parameters())

            for p, v in zip(self.model_2.parameters(), vector_2):
                p.data.add_(R, v)

            term_2 = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

            return term_1+term_2
        else:
            R = r / _concat(vector_1).norm()
            for p, v in zip(self.model_1.parameters(), vector_1):
                p.data.add_(R, v)
            loss = self._get_loss_val(self, input, target)
            grads_p = torch.autograd.grad(loss, self.model_2.arch_parameters())

            for p, v in zip(self.model_1.parameters(), vector_1):
                p.data.sub_(2*R, v)
            loss = self._get_loss_val(self, input, target)
            grads_n = torch.autograd.grad(loss, self.model_2.arch_parameters())

            for p, v in zip(self.model_1.parameters(), vector_1):
                p.data.add_(R, v)

            term_1 = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

            R = r / _concat(vector_2).norm()
            for p, v in zip(self.model_2.parameters(), vector_2):
                p.data.add_(R, v)
            loss = self._get_loss_val(self, input, target)
            grads_p = torch.autograd.grad(loss, self.model_2.arch_parameters())

            for p, v in zip(self.model_2.parameters(), vector_2):
                p.data.sub_(2*R, v)
            loss = self._get_loss_val(self, input, target)
            grads_n = torch.autograd.grad(loss, self.model_2.arch_parameters())

            for p, v in zip(self.model_2.parameters(), vector_2):

                p.data.add_(R, v)

            term_2 = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

            return term_1+term_2

    def _get_loss_val(self, input, target):
        loss_1, logits_1 = self.model_1._loss(input, target)
        loss_2, logits_2 = self.model_2._loss(input, target)

        logits_3 = torch.log10(F.softmax(self.model_1(input), dim=1))
        logits_4 = torch.log(F.softmax(self.model_2(input), dim=1))

        logits_2 = F.softmax(logits_2, dim=1)
        logits_1 = F.softmax(logits_1, dim=1)

        loss = torch.sum(logits_2*logits_3*-1) + \
            torch.sum(logits_1*logits_4*-1)

        # loss_3 = cross_entropy_loss_softmax(input, logits_2, self.model_1)
        # loss_4 = cross_entropy_loss_softmax(input, logits_1, self.model_2)

        loss = (loss_1 + loss_2) + self.c_lambda * (loss)
        print("Term 1 = " + str((loss_1 + loss_2)))
        print("Term 2 = " + str(loss))

        return loss


def cross_entropy_loss_softmax(input, labels_fake, model):
    logits = model(input)
    logits = torch.log(logits)
    print(logits)
    print(labels_fake)
    loss = torch.sum(logits*labels_fake*-1)
    print(loss)
    return loss
