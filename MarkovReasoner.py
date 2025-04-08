import numpy as np
from typing import Dict, List, Tuple

class MarkovReasoner:
    def __init__(self):
        """Inicializa o resolvedor de problemas com cadeias de Markov"""
        self.states = {}  # Dicionário de estados possíveis
        self.transitions = {}  # Matriz de transição entre estados
        self.problem = None  # Problema atual
        self.solution_path = []  # Caminho da solução
        
    def add_state(self, state_name: str, description: str, is_solution: bool = False):
        """Adiciona um estado ao espaço de raciocínio"""
        self.states[state_name] = {
            'description': description,
            'is_solution': is_solution,
            'index': len(self.states)
        }
        
    def add_transition(self, from_state: str, to_state: str, probability: float):
        """Define uma transição entre estados com certa probabilidade"""
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        self.transitions[from_state][to_state] = probability
        
    def normalize_transitions(self):
        """Normaliza as probabilidades de transição para somarem 1"""
        for from_state in self.transitions:
            total = sum(self.transitions[from_state].values())
            for to_state in self.transitions[from_state]:
                self.transitions[from_state][to_state] /= total
                
    def solve_problem(self, initial_state: str, max_steps: int = 100) -> List[str]:
        """Tenta resolver o problema começando do estado inicial"""
        self.solution_path = [initial_state]
        current_state = initial_state
        
        for _ in range(max_steps):
            if self.states[current_state]['is_solution']:
                return self.solution_path
                
            # Escolhe próximo estado baseado nas probabilidades de transição
            next_state = np.random.choice(
                list(self.transitions.get(current_state, {}).keys()),
                p=list(self.transitions.get(current_state, {}).values())
            )
            
            self.solution_path.append(next_state)
            current_state = next_state
            
        return self.solution_path  # Retorna mesmo se não encontrar solução
    
    def print_solution_path(self):
        """Imprime o caminho percorrido na tentativa de solução"""
        print("Caminho de raciocínio:")
        for i, state in enumerate(self.solution_path):
            print(f"{i+1}. {self.states[state]['description']}")
            if self.states[state]['is_solution']:
                print("→ Solução encontrada!")
                break