use proc_macro::{self, TokenStream};
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

macro_rules! binary_op_derive {
    ($derive:ident, $op:ident, $func:ident) => {
        #[proc_macro_derive($derive)]
        pub fn $func(input: TokenStream) -> TokenStream {
            let DeriveInput {
                ident, generics, ..
            } = parse_macro_input!(input);
            let type_params = generics.type_params();
            let (_, ty_generics, where_clause) = generics.split_for_impl();
            let predicates = where_clause.map_or(
                quote! {},
                |syn::WhereClause { predicates, .. }| quote! { #predicates },
            );
            let output = quote! {
                #[automatically_derived]
                impl <#(#type_params),*,
                    O: crate::MatrixShape<Scalar = <Self as crate::MatrixShape>::Scalar>>
                std::ops::$op<O> for #ident #ty_generics
                where
                    <Self as crate::MatrixShape>::Scalar: std::ops::$op<
                        <Self as crate::MatrixShape>::Scalar,
                        Output=<Self as crate::MatrixShape>::Scalar,
                    >,
                    #predicates
                {
                    type Output = crate::cwise_ops::CwiseBinaryOp<
                        crate::cwise_ops::$op<<Self as crate::MatrixShape>::Scalar>, Self, O>;
                    fn $func(self, other: O) -> Self::Output {
                        crate::cwise_ops::CwiseBinaryOp::new(self, other)
                    }
                }
            };
            output.into()
        }
    };
}

binary_op_derive!(CwiseAdd, Add, add);
binary_op_derive!(CwiseSub, Sub, sub);
binary_op_derive!(CwiseMul, Mul, mul);

#[proc_macro_derive(CwiseNeg)]
pub fn neg(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident, generics, ..
    } = parse_macro_input!(input);
    let (ty_impl, ty_generics, where_clause) = generics.split_for_impl();
    let predicates = where_clause.map_or(
        quote! {},
        |syn::WhereClause { predicates, .. }| quote! { #predicates },
    );
    let output = quote! {
        #[automatically_derived]
        impl #ty_impl std::ops::Neg for #ident #ty_generics
        where
            <Self as crate::MatrixShape>::Scalar: std::ops::Neg<
                Output=<Self as crate::MatrixShape>::Scalar,
            >,
            #predicates
        {
            type Output = crate::cwise_ops::CwiseUnaryOp<
                crate::cwise_ops::Neg<<Self as crate::MatrixShape>::Scalar>, Self>;
            fn neg(self) -> Self::Output {
                crate::cwise_ops::CwiseUnaryOp::new(self)
            }
        }
    };
    output.into()
}
