use fastset::Set;
use itertools::Itertools;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

#[inline(always)]
fn codebook_contains(
    codebook: &FxHashMap<Vec<usize>, usize>,
    ids: &Vec<usize>,
    initial_vocab_size: usize,
) -> bool {
    if ids.len() == 1 {
        ids[0] < initial_vocab_size
    } else {
        codebook.contains_key(ids)
    }
}

#[inline(always)]
fn get_usize_from_codebook(codebook: &FxHashMap<Vec<usize>, usize>, ids: &Vec<usize>) -> usize {
    if ids.len() == 1 {
        ids[0]
    } else {
        codebook.get(ids).unwrap().clone()
    }
}

#[inline(always)]
fn disabled_ids_to_set(disabled_ids: Option<Vec<usize>>) -> Set {
    disabled_ids.map_or_else(
        || Set::with_capacity(0),
        |d_ids| {
            let mut set = Set::with_capacity(d_ids.iter().max().unwrap() + 1);
            for id in d_ids {
                set.insert(id);
            }
            set
        },
    )
}

fn compress(
    ids: &Vec<usize>,
    initial_vocab_size: usize,
    max_codebook_size: usize,
    max_subtokens: usize,
    max_out_seq_length: usize,
    eot_token_id: usize,
    disabled_ids: &Set,
) -> (Vec<usize>, Vec<Vec<usize>>, Option<Vec<usize>>) {
    let mut compressed_ids: Vec<usize> = Vec::new();
    let mut codebook: FxHashMap<Vec<usize>, usize> = FxHashMap::default();

    let mut next_id: usize = initial_vocab_size;
    let mut ids_to_merge: Vec<usize> = Vec::with_capacity(max_subtokens);

    let mut i = 0;
    while i < ids.len() && i < max_out_seq_length {
        let id = ids[i];

        if disabled_ids.contains(&id) {
            if ids_to_merge.len() > 0 {
                compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
                ids_to_merge.clear();
            }
            compressed_ids.push(id);
            continue;
        }

        ids_to_merge.push(id);

        let is_in_codebook = codebook_contains(&codebook, &ids_to_merge, initial_vocab_size);
        if !is_in_codebook {
            if next_id < initial_vocab_size + max_codebook_size {
                codebook.insert(ids_to_merge.clone(), next_id);
                next_id += 1;
            }

            ids_to_merge.pop();
            compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
            ids_to_merge.clear();
            ids_to_merge.push(id);
        }

        if ids_to_merge.len() == max_subtokens {
            compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
            ids_to_merge.clear();
        }

        i += 1;
    }

    if ids_to_merge.len() > max_subtokens {
        let last_id = ids_to_merge.pop().unwrap();
        compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
        ids_to_merge.clear();
        ids_to_merge.push(last_id);
    }

    if ids_to_merge.len() > 0 {
        compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
    }

    let mut codebook_vec: Vec<Vec<usize>> = codebook
        .iter()
        .sorted_by(|(_, value), (_, value2)| value.cmp(value2))
        .map(|(key, _)| {
            let mut k = key.clone();
            k.resize(max_subtokens, eot_token_id);
            k
        })
        .collect::<Vec<Vec<usize>>>();

    codebook_vec.resize(max_codebook_size, vec![eot_token_id; max_subtokens]);

    let mut j = i;
    let mut has_remaining_docs = false;
    while j < ids.len() {
        if ids[j] == eot_token_id {
            has_remaining_docs = true;
            break;
        }
        j += 1;
    }

    let remaining_ids = if has_remaining_docs {
        Some(ids[j..].to_vec())
    } else {
        None
    };

    (compressed_ids, codebook_vec, remaining_ids)
}

#[pyfunction]
#[pyo3(name = "compress")]
fn py_compress(
    ids: Vec<usize>,
    initial_vocab_size: usize,
    max_codebook_size: usize,
    max_subtokens: usize,
    max_out_seq_length: usize,
    eot_token_id: usize,
    disabled_ids: Option<Vec<usize>>,
) -> PyResult<(Vec<usize>, Vec<Vec<usize>>, Option<Vec<usize>>)> {
    let (compressed_ids, codebook_vec, remaining_ids) = compress(
        &ids,
        initial_vocab_size,
        max_codebook_size,
        max_subtokens,
        max_out_seq_length,
        eot_token_id,
        &disabled_ids_to_set(disabled_ids),
    );

    Ok((compressed_ids, codebook_vec, remaining_ids))
}

#[pymodule]
fn fast_compression(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_compress, m)?)?;

    Ok(())
}
