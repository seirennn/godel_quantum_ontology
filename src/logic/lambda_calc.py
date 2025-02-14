"""
Lambda calculus implementation for the Divine Algorithm project.
"""

from typing import Dict, List, Optional, Set, Any
from abc import ABC, abstractmethod

class Term(ABC):
    """Abstract base class for lambda calculus terms."""
    
    @abstractmethod
    def __str__(self) -> str:
        """String representation of the term."""
        pass
    
    @abstractmethod
    def free_variables(self) -> Set[str]:
        """Get set of free variables in the term."""
        pass
    
    @abstractmethod
    def substitute(self, var: str, term: "Term") -> "Term":
        """Substitute term for variable in this term."""
        if not isinstance(term, Term):
            raise TypeError("Substitution term must be a Term")
        pass
    
    @abstractmethod
    def evaluate(self) -> "Term":
        """Evaluate term to normal form."""
        pass
    
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Compare terms for equality."""
        pass

class Variable(Term):
    """Variable term in lambda calculus."""
    
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self) -> str:
        return self.name
    
    def free_variables(self) -> Set[str]:
        return {self.name}
    
    def substitute(self, var: str, term: Term) -> Term:
        if not isinstance(term, Term):
            raise TypeError("Substitution term must be a Term")
        if var == self.name:
            return term
        return self
    
    def evaluate(self) -> Term:
        return self
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Variable):
            return False
        return self.name == other.name

class Abstraction(Term):
    """Lambda abstraction (function) term."""
    
    def __init__(self, var: str, body: Term):
        if not isinstance(body, Term):
            raise TypeError("Body must be a Term")
        self.var = var
        self.body = body
    
    def __str__(self) -> str:
        return f"λ{self.var}.{str(self.body)}"
    
    def free_variables(self) -> Set[str]:
        return self.body.free_variables() - {self.var}
    
    def substitute(self, var: str, term: Term) -> Term:
        if not isinstance(term, Term):
            raise TypeError("Substitution term must be a Term")
        if var == self.var:
            return self
        if var not in term.free_variables():
            return Abstraction(self.var, self.body.substitute(var, term))
        # Alpha conversion needed
        new_var = self._fresh_var(term.free_variables())
        new_body = self.body.substitute(self.var, Variable(new_var))
        return Abstraction(new_var, new_body.substitute(var, term))
    
    def evaluate(self) -> Term:
        return Abstraction(self.var, self.body.evaluate())
    
    def _fresh_var(self, used_vars: Set[str]) -> str:
        """Generate fresh variable name."""
        base = self.var
        i = 0
        while f"{base}{i}" in used_vars:
            i += 1
        return f"{base}{i}"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Abstraction):
            return False
        return self.var == other.var and self.body == other.body

class Application(Term):
    """Function application term."""
    
    def __init__(self, func: Term, arg: Term):
        if not isinstance(func, Term) or not isinstance(arg, Term):
            raise TypeError("Function and argument must be Terms")
        self.func = func
        self.arg = arg
    
    def __str__(self) -> str:
        func_str = str(self.func)
        arg_str = str(self.arg)
        if isinstance(self.arg, Application):
            arg_str = f"({arg_str})"
        return f"{func_str} {arg_str}"
    
    def free_variables(self) -> Set[str]:
        return self.func.free_variables() | self.arg.free_variables()
    
    def substitute(self, var: str, term: Term) -> Term:
        if not isinstance(term, Term):
            raise TypeError("Substitution term must be a Term")
        return Application(
            self.func.substitute(var, term),
            self.arg.substitute(var, term)
        )
    
    def evaluate(self) -> Term:
        func = self.func.evaluate()
        arg = self.arg.evaluate()
        if isinstance(func, Abstraction):
            return func.body.substitute(func.var, arg).evaluate()
        return Application(func, arg)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Application):
            return False
        return self.func == other.func and self.arg == other.arg

class LambdaCalculus:
    """Lambda calculus implementation."""
    
    def __init__(self):
        self.church_numerals = {}
        self._init_church_numerals()
    
    def _init_church_numerals(self):
        """Initialize Church numerals 0-5."""
        for i in range(6):
            self.church_numerals[i] = self._create_church_numeral(i)
    
    def create_church_numeral(self, n: int) -> Term:
        """Create Church numeral for given integer."""
        if not isinstance(n, int):
            raise TypeError("Church numeral must be an integer")
        if n < 0:
            raise ValueError("Church numerals must be non-negative")
        return self._create_church_numeral(n)
    
    def create_boolean(self, value: bool) -> Term:
        """Create Church boolean for given value."""
        if not isinstance(value, bool):
            raise TypeError("Value must be a boolean")
        if value:
            return Abstraction("t", Abstraction("f", Variable("t")))  # true = λt.λf.t
        else:
            return Abstraction("t", Abstraction("f", Variable("f")))  # false = λt.λf.f
    
    def _create_church_numeral(self, n: int) -> Term:
        """Create Church numeral for given integer."""
        var_f = Variable("f")
        var_x = Variable("x")
        body = var_x
        for _ in range(n):
            body = Application(var_f, body)
        return Abstraction("f", Abstraction("x", body))

class QuantumLogicAnalyzer:
    """Analyzer for quantum logic using lambda calculus."""
    
    def __init__(self):
        self.calculus = LambdaCalculus()
    
    def analyze_quantum_logic(self, term: Optional[Term]) -> Dict[str, Any]:
        """
        Analyze quantum logic term.
        
        Args:
            term: Lambda calculus term to analyze
            
        Returns:
            Dictionary of analysis results
            
        Raises:
            AttributeError: If term is None
            TypeError: If term is not a Term instance
        """
        # Let None raise AttributeError naturally
        term.free_variables()  # This will raise AttributeError if term is None
        
        if not isinstance(term, Term):
            raise TypeError("Input must be a Term")
            
        analysis = {}
        
        # Analyze term structure
        analysis["free_vars"] = list(term.free_variables())
        analysis["complexity"] = self._calculate_complexity(term)
        
        # Evaluate term
        try:
            evaluated = term.evaluate()
            analysis["evaluated"] = str(evaluated)
            analysis["reducible"] = str(evaluated) != str(term)
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def _calculate_complexity(self, term: Term) -> int:
        """Calculate complexity score for term."""
        if isinstance(term, Variable):
            return 1
        if isinstance(term, Abstraction):
            return 1 + self._calculate_complexity(term.body)
        if isinstance(term, Application):
            return (
                1 +
                self._calculate_complexity(term.func) +
                self._calculate_complexity(term.arg)
            )
        return 0
