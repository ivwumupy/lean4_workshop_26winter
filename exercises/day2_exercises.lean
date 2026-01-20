import Mathlib

-- 请为以下的examples起一个合乎规范的名字（其实你们多数也见过了）：

variable {α : Type}

example {a : ℝ} : a + 0 = 0 := by sorry

example {a b c : ℝ} : (a + b) * c = a * c + b * c := by sorry

example {a b : ℝ} : a / b = a * b⁻¹ := by sorry

example {a b c : ℝ} : a ∣ b - c → (a ∣ b ↔ a ∣ c) := by sorry

example (s t : Set α) (x : α) : x ∈ s → x ∈ s ∪ t := by sorry

example (s t : Set α) (x : α) : x ∈ s ∪ t → x ∈ s ∨ x ∈ t := by sorry

example {a : α} {p : α → Prop} : a ∈ {x | p x} ↔ p a := by sorry

example {x a : α} {s : Set α} : x ∈ insert a s → x = a ∨ x ∈ s := by sorry

example {x : α} {a b : Set α} : x ∈ a ∩ b → x ∈ a := by sorry

example {a b : ℝ} : a ≤ b ↔ a < b ∨ a = b := by sorry

example {a b : ℤ} : a ≤ b - 1 ↔ a < b := by sorry

example {a b c : ℝ} (bc : a + b ≤ a + c) : b ≤ c := by sorry



-- 请根据以下命名猜测并陈述出定理

/-
1. mul_add
2. add_sub_right_comm
3. le_of_lt_of_le
4. add_le_add
5. mem_union_of_mem_right
-/


-- tactics练习
variable (a b c d : ℝ)

#check pow_eq_zero
#check pow_two_nonneg
example {x y : ℝ} (h : x ^ 2 + y ^ 2 = 0) : x = 0 := by sorry

theorem aux : min a b + c ≤ min (a + c) (b + c) := by sorry

-- 你可以尝试使用aux来完成这一证明
#check le_antisymm
#check add_le_add_right
#check sub_eq_add_neg
example : min a b + c = min (a + c) (b + c) := by sorry

#check sq_nonneg
theorem fact1 : a * b * 2 ≤ a ^ 2 + b ^ 2 := by sorry

#check pow_two_nonneg
theorem fact2 : -(a * b) * 2 ≤ a ^ 2 + b ^ 2 := by sorry

-- 你可以使用上面两个定理来完成这一证明
#check le_div_iff₀
example : |a * b| ≤ (a ^ 2 + b ^ 2) / 2 := by sorry

-- Finish the proof using the theorems abs_mul, mul_le_mul, abs_nonneg, mul_lt_mul_right, and one_mul.
theorem my_lemma4 :
    ∀ {x y ε : ℝ}, 0 < ε → ε ≤ 1 → |x| < ε → |y| < ε → |x * y| < ε := by
  intro x y ε epos ele1 xlt ylt
  calc
    |x * y| = |x| * |y| := sorry
    _ ≤ |x| * ε := sorry
    _ < 1 * ε := sorry
    _ = ε := sorry



def FnUb (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x ≤ a

def FnLb (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, a ≤ f x

section
variable (f g : ℝ → ℝ) (a b : ℝ)


example (hfa : FnLb f a) (hgb : FnLb g b) : FnLb (fun x ↦ f x + g x) (a + b) := by
  sorry

example (hfa : FnUb f a) (hgb : FnUb g b) (nng : FnLb g 0) (nna : 0 ≤ a) :
    FnUb (fun x ↦ f x * g x) (a * b) := by
  sorry


-- 使用calc
example (a b : ℝ) : - 2 * a * b ≤ a ^ 2 + b ^ 2 := by sorry

example (a b c : ℝ) : a * b + a * c + b * c ≤ a * a + b * b + c * c := by sorry

example {x y ε : ℝ} (epos : 0 < ε) (ele1 : ε ≤ 1) (xlt : |x| < ε) (ylt : |y| < ε) : |x * y| < ε := by sorry

end
