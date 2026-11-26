var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __knownSymbol = (name, symbol) => (symbol = Symbol[name]) ? symbol : Symbol.for("Symbol." + name);
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var __await = function(promise, isYieldStar) {
  this[0] = promise;
  this[1] = isYieldStar;
};
var __asyncGenerator = (__this, __arguments, generator) => {
  var resume = (k, v, yes, no) => {
    try {
      var x = generator[k](v), isAwait = (v = x.value) instanceof __await, done = x.done;
      Promise.resolve(isAwait ? v[0] : v).then((y) => isAwait ? resume(k === "return" ? k : "next", v[1] ? { done: y.done, value: y.value } : y, yes, no) : yes({ value: y, done })).catch((e) => resume("throw", e, yes, no));
    } catch (e) {
      no(e);
    }
  }, method = (k) => it[k] = (x) => new Promise((yes, no) => resume(k, x, yes, no)), it = {};
  return generator = generator.apply(__this, __arguments), it[__knownSymbol("asyncIterator")] = () => it, method("next"), method("throw"), method("return"), it;
};
var __forAwait = (obj, it, method) => (it = obj[__knownSymbol("asyncIterator")]) ? it.call(obj) : (obj = obj[__knownSymbol("iterator")](), it = {}, method = (key, fn) => (fn = obj[key]) && (it[key] = (arg) => new Promise((yes, no, done) => (arg = fn.call(obj, arg), done = arg.done, Promise.resolve(arg.value).then((value) => yes({ value, done }), no)))), method("next"), method("return"), it);

// src/glue/messages.ts
var GLUE_VERSION = 1;
var GLUE_MESSAGE_PROTOTYPES = {
  "erro_evt": {
    "name": "erro_evt",
    "structName": "glue_msg_error",
    "className": "GlueMsgError",
    "fields": [
      {
        "type": "str",
        "name": "message",
        "isNullable": false
      }
    ]
  },
  "load_req": {
    "name": "load_req",
    "structName": "glue_msg_load_req",
    "className": "GlueMsgLoadReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "model_paths",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "mmproj_path",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "n_ctx_auto",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mmap",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mlock",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_gpu_layers",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_threads",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "model_alias",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "log_level",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "embeddings",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "offload_kqv",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_parallel",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "pooling_type",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "rope_scaling_type",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_base",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_scale",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_ext_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_attn_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_fast",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_slow",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "yarn_orig_ctx",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_k",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_v",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "kv_unified",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "flash_attn",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "swa_full",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ctx_checkpoints",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "checkpoint_min_step",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "chat_template",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "jinja",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_vals",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "reasoning",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_min_tokens",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_max_tokens",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "warmup",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "no_kv_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "mmproj_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cont_batching",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_keep",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "ctx_shift",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cache_idle_slots",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_cache_reuse",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "lora_paths",
        "isNullable": true
      },
      {
        "type": "arr_float",
        "name": "lora_scales",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "lora_init_without_apply",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "native_state_paths",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "spec_draft_model",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_ngl",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_max",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_min",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "spec_draft_p_min",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads_batch",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_vals",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "reasoning_budget_tokens",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_budget_message",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_format",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "skip_chat_parsing",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "prefill_assistant",
        "isNullable": true
      }
    ]
  },
  "load_res": {
    "name": "load_res",
    "structName": "glue_msg_load_res",
    "className": "GlueMsgLoadRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_vocab",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx_train",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_embd",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_layer",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_key",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_val",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_bos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eot",
        "isNullable": false
      },
      {
        "type": "arr_int",
        "name": "list_tokens_eog",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_bos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_eos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_encoder",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_decoder_start",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "media_marker",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_image_input",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_audio_input",
        "isNullable": false
      }
    ]
  },
  "cmpl_req": {
    "name": "cmpl_req",
    "structName": "glue_msg_completion_req",
    "className": "GlueMsgCompletionReq",
    "fields": [
      {
        "type": "bool",
        "name": "is_chat",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "cmpl_res": {
    "name": "cmpl_res",
    "structName": "glue_msg_completion_res",
    "className": "GlueMsgCompletionRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "embd_req": {
    "name": "embd_req",
    "structName": "glue_msg_embedding_req",
    "className": "GlueMsgEmbeddingReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "embd_res": {
    "name": "embd_res",
    "structName": "glue_msg_embedding_res",
    "className": "GlueMsgEmbeddingRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "rrnk_req": {
    "name": "rrnk_req",
    "structName": "glue_msg_rerank_req",
    "className": "GlueMsgRerankReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "rrnk_res": {
    "name": "rrnk_res",
    "structName": "glue_msg_rerank_res",
    "className": "GlueMsgRerankRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "sys1_req": {
    "name": "sys1_req",
    "structName": "glue_msg_systemone_req",
    "className": "GlueMsgSystemoneReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "sys1_res": {
    "name": "sys1_res",
    "structName": "glue_msg_systemone_res",
    "className": "GlueMsgSystemoneRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "scor_req": {
    "name": "scor_req",
    "structName": "glue_msg_score_req",
    "className": "GlueMsgScoreReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "scor_res": {
    "name": "scor_res",
    "structName": "glue_msg_score_res",
    "className": "GlueMsgScoreRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "slot_req": {
    "name": "slot_req",
    "structName": "glue_msg_slots_req",
    "className": "GlueMsgSlotsReq",
    "fields": [
      {
        "type": "int",
        "name": "id_slot",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "slot_res": {
    "name": "slot_res",
    "structName": "glue_msg_slots_res",
    "className": "GlueMsgSlotsRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "nsta_req": {
    "name": "nsta_req",
    "structName": "glue_msg_native_state_req",
    "className": "GlueMsgNativeStateReq",
    "fields": [
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "id",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "nsta_res": {
    "name": "nsta_res",
    "structName": "glue_msg_native_state_res",
    "className": "GlueMsgNativeStateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "meta_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_req": {
    "name": "tmpl_req",
    "structName": "glue_msg_apply_template_req",
    "className": "GlueMsgApplyTemplateReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_res": {
    "name": "tmpl_res",
    "structName": "glue_msg_apply_template_res",
    "className": "GlueMsgApplyTemplateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "prompt",
        "isNullable": false
      }
    ]
  },
  "gres_req": {
    "name": "gres_req",
    "structName": "glue_msg_get_result_req",
    "className": "GlueMsgGetResultReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_res": {
    "name": "gres_res",
    "structName": "glue_msg_get_result_res",
    "className": "GlueMsgGetResultRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_more",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "is_error",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "cncl_req": {
    "name": "cncl_req",
    "structName": "glue_msg_cancel_req",
    "className": "GlueMsgCancelReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "cncl_res": {
    "name": "cncl_res",
    "structName": "glue_msg_cancel_res",
    "className": "GlueMsgCancelRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  },
  "tbop_req": {
    "name": "tbop_req",
    "structName": "glue_msg_test_backend_ops_req",
    "className": "GlueMsgTestBackendOpsReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "args",
        "isNullable": false
      }
    ]
  },
  "tbop_res": {
    "name": "tbop_res",
    "structName": "glue_msg_test_backend_ops_res",
    "className": "GlueMsgTestBackendOpsRes",
    "fields": [
      {
        "type": "int",
        "name": "retcode",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  }
};

// src/glue/glue.ts
var GLUE_MAGIC = new Uint8Array([71, 76, 85, 69]);
var GLUE_DTYPE_NULL = 0;
var GLUE_DTYPE_BOOL = 1;
var GLUE_DTYPE_INT = 2;
var GLUE_DTYPE_FLOAT = 3;
var GLUE_DTYPE_STRING = 4;
var GLUE_DTYPE_RAW = 5;
var GLUE_DTYPE_ARRAY_BOOL = 6;
var GLUE_DTYPE_ARRAY_INT = 7;
var GLUE_DTYPE_ARRAY_FLOAT = 8;
var GLUE_DTYPE_ARRAY_STRING = 9;
var GLUE_DTYPE_ARRAY_RAW = 10;
var TYPE_MAP = {
  str: GLUE_DTYPE_STRING,
  int: GLUE_DTYPE_INT,
  float: GLUE_DTYPE_FLOAT,
  bool: GLUE_DTYPE_BOOL,
  raw: GLUE_DTYPE_RAW,
  arr_str: GLUE_DTYPE_ARRAY_STRING,
  arr_int: GLUE_DTYPE_ARRAY_INT,
  arr_float: GLUE_DTYPE_ARRAY_FLOAT,
  arr_bool: GLUE_DTYPE_ARRAY_BOOL,
  arr_raw: GLUE_DTYPE_ARRAY_RAW,
  null: GLUE_DTYPE_NULL
};
function glueDeserialize(buf) {
  let offset = 0;
  const view = new DataView(buf.buffer);
  const readUint32 = () => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readInt32 = () => {
    const value = view.getInt32(offset, true);
    offset += 4;
    return value;
  };
  const readFloat = () => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };
  const readBool = () => {
    return readUint32() !== 0;
  };
  const readString = (customLen) => {
    const length = customLen != null ? customLen : readUint32();
    const value = new TextDecoder().decode(buf.slice(offset, offset + length));
    offset += length;
    return value;
  };
  const readRaw = () => {
    const length = readUint32();
    const value = buf.slice(offset, offset + length);
    offset += length;
    return value;
  };
  const readArray = (readItem) => {
    const length = readUint32();
    const value = new Array(length);
    for (let i = 0; i < length; i++) {
      value[i] = readItem();
    }
    return value;
  };
  const readNull = () => null;
  const readField = (field) => {
    switch (field.type) {
      case "str":
        return readString();
      case "int":
        return readInt32();
      case "float":
        return readFloat();
      case "bool":
        return readBool();
      case "raw":
        return readRaw();
      case "arr_str":
        return readArray(readString);
      case "arr_int":
        return readArray(readInt32);
      case "arr_float":
        return readArray(readFloat);
      case "arr_bool":
        return readArray(readBool);
      case "arr_raw":
        return readArray(readRaw);
      case "null":
        return readNull();
    }
  };
  const magicValid = buf[0] === GLUE_MAGIC[0] && buf[1] === GLUE_MAGIC[1] && buf[2] === GLUE_MAGIC[2] && buf[3] === GLUE_MAGIC[3];
  offset += 4;
  if (!magicValid) {
    throw new Error("Invalid magic number");
  }
  const version = readUint32();
  if (version !== GLUE_VERSION) {
    throw new Error("Invalid version number");
  }
  const name = readString(8);
  const msgProto = GLUE_MESSAGE_PROTOTYPES[name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${name}`);
  }
  const output = { _name: name };
  for (const field of msgProto.fields) {
    const readType = readUint32();
    if (readType === GLUE_DTYPE_NULL) {
      if (!field.isNullable) {
        throw new Error(
          `${name}: Expect field ${field.name} to be non-nullable`
        );
      }
      output[field.name] = null;
      continue;
    }
    if (readType !== TYPE_MAP[field.type]) {
      throw new Error(
        `${name}: Expect field ${field.name} to have type ${field.type}`
      );
    }
    output[field.name] = readField(field);
  }
  return output;
}
function glueSerialize(msg) {
  const msgProto = GLUE_MESSAGE_PROTOTYPES[msg._name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${msg._name}`);
  }
  const bufs = [];
  const writeUint32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeInt32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setInt32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeFloat = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeBool = (value) => {
    writeUint32(value ? 1 : 0);
  };
  const writeString = (value) => {
    const utf8 = new TextEncoder().encode(value);
    writeUint32(utf8.byteLength);
    bufs.push(utf8);
  };
  const writeRaw = (value) => {
    writeUint32(value.byteLength);
    bufs.push(value);
  };
  const writeArray = (value, writeItem) => {
    writeUint32(value.length);
    for (const item of value) {
      writeItem(item);
    }
  };
  const writeNull = () => {
  };
  bufs.push(GLUE_MAGIC);
  writeUint32(GLUE_VERSION);
  {
    const utf8 = new TextEncoder().encode(msg._name);
    bufs.push(utf8);
  }
  for (const field of msgProto.fields) {
    const val = msg[field.name];
    if (!field.isNullable && (val === null || val === void 0)) {
      throw new Error(
        `${msg._name}: Expect field ${field.name} to be non-nullable`
      );
    }
    if (val === null || val === void 0) {
      writeUint32(GLUE_DTYPE_NULL);
      continue;
    }
    writeUint32(TYPE_MAP[field.type]);
    switch (field.type) {
      case "str":
        writeString(val);
        break;
      case "int":
        writeInt32(val);
        break;
      case "float":
        writeFloat(val);
        break;
      case "bool":
        writeBool(val);
        break;
      case "raw":
        writeRaw(val);
        break;
      case "arr_str":
        writeArray(val, writeString);
        break;
      case "arr_int":
        writeArray(val, writeInt32);
        break;
      case "arr_float":
        writeArray(val, writeFloat);
        break;
      case "arr_bool":
        writeArray(val, writeBool);
        break;
      case "arr_raw":
        writeArray(val, writeRaw);
        break;
      case "null":
        writeNull();
        break;
    }
  }
  const totalLength = bufs.reduce((acc, buf) => acc + buf.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const buf of bufs) {
    output.set(buf, offset);
    offset += buf.byteLength;
  }
  return output;
}

// src/wasm/source-map.ts
var WASM_SOURCE_MAP = {
  "default": "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",
  "compat": "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"
};

// src/debug.ts
var cache = /* @__PURE__ */ new Map();
function loadMap(buildKey) {
  return __async(this, null, function* () {
    if (cache.has(buildKey)) return cache.get(buildKey);
    const b64 = WASM_SOURCE_MAP[buildKey];
    if (!b64) throw new Error(`No source map for build "${buildKey}"`);
    const gzipped = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const ds = new DecompressionStream("gzip");
    const writer = ds.writable.getWriter();
    writer.write(gzipped);
    writer.close();
    const buf = yield new Response(ds.readable).arrayBuffer();
    const dv = new DataView(buf);
    const bytes = new Uint8Array(buf);
    const firstId = dv.getUint32(0, true);
    const funcCount = dv.getUint32(4, true);
    const numNames = dv.getUint32(8, true);
    const td = new TextDecoder();
    const names = [];
    let pos = 12;
    for (let i = 0; i < numNames; i++) {
      const len = bytes[pos++];
      names.push(td.decode(bytes.subarray(pos, pos + len)));
      pos += len;
    }
    const funcNames = [];
    for (let i = 0; i < funcCount; i++) {
      const idx = dv.getUint16(pos, true);
      pos += 2;
      funcNames.push(idx === 65535 ? null : names[idx]);
    }
    const entry = { firstId, funcNames };
    cache.set(buildKey, entry);
    return entry;
  });
}
var Debug = {
  /**
   * Resolves a list of wasm function indices to their cleaned symbol names.
   */
  decodeFuncIds: (funcIds, isCompatBuild) => __async(void 0, null, function* () {
    const buildKey = isCompatBuild ? "compat" : "default";
    const { firstId, funcNames } = yield loadMap(buildKey);
    return funcIds.map((funcId) => {
      const i = funcId - firstId;
      const name = i >= 0 && i < funcNames.length && funcNames[i] ? funcNames[i] : "(unknown)";
      return { funcId, name };
    });
  }),
  /**
   * Annotates a wasm stack trace string with resolved function names.
   *
   * Example input from Chrome:
   *   at http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[775]:0x74251
   *   at async blob:http://localhost:8080/53a863cc-7227-45cc-8594-ddbbf5257f20:317:28
   *
   * Example input from Firefox:
   *   @http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[796]:0x7dfe2
   *       at wModuleInit/WebAssembly.promising/< (9b6a2acd-d909-44e2-b021-d42fb9087cfb:15:32) index.js:1433:45
   *
   * Example input from Safari:
   *   2441@wasm-function[2441]
   *       at wrapper (d746f19e-4523-4f36-ba06-d0969acc0b05:22:126009)
   *
   * Example output:
   *   wasm-func[775] (server_response::send)
   */
  decodeStackTrace: (stack, isCompatBuild) => __async(void 0, null, function* () {
    const re = /wasm-function\[(\d+)\]/g;
    const funcIds = [
      ...new Set([...stack.matchAll(re)].map((m) => parseInt(m[1])))
    ];
    if (funcIds.length === 0) return stack;
    const resolved = yield Debug.decodeFuncIds(funcIds, isCompatBuild);
    return resolved.map((r) => {
      if (r.name === "(unknown)") {
        return `    wasm-func[${r.funcId}] (unknown)`;
      }
      return `    wasm-func[${r.funcId}] (${r.name})`;
    }).join("\n");
  })
};

// src/utils.ts
var textDecoder = new TextDecoder();
var URL_PARTS_REGEX = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
var parseShardNumber = (fnameOrUrl) => {
  const matches = fnameOrUrl.match(URL_PARTS_REGEX);
  if (!matches) {
    return {
      baseURL: fnameOrUrl,
      current: 1,
      total: 1
    };
  } else {
    return {
      baseURL: fnameOrUrl.replace(URL_PARTS_REGEX, ""),
      current: parseInt(matches[1]),
      total: parseInt(matches[2])
    };
  }
};
var sortFileByShard = (blobs) => {
  const isFiles = blobs.every((b) => !!b.name);
  if (isFiles && blobs.length > 1) {
    const files = blobs;
    files.sort((a, b) => {
      const infoA = parseShardNumber(a.name);
      const infoB = parseShardNumber(b.name);
      return infoA.current - infoB.current;
    });
  }
};
var isMmproj = (blob) => __async(void 0, null, function* () {
  const META_NAME = "general.architecture";
  const META_VAL = "clip";
  const tmp = blob.slice(0, 128 * 1024);
  const header = yield tmp.arrayBuffer();
  const buf = new Uint8Array(header);
  const nameBytes = new TextEncoder().encode(META_NAME);
  const valBytes = new TextEncoder().encode(META_VAL);
  let offset = -1;
  outer: for (let i = 0; i <= buf.length - nameBytes.length; i++) {
    for (let j = 0; j < nameBytes.length; j++) {
      if (buf[i + j] !== nameBytes[j]) continue outer;
    }
    offset = i;
    break;
  }
  if (offset === -1) return false;
  if (offset + 8 * 4 + 4 > buf.length) return false;
  const view = new DataView(header);
  const valLen = view.getBigUint64(offset + 8 * 3, true);
  if (valLen !== /* @__PURE__ */ BigInt("4")) return false;
  for (let i = 0; i < valBytes.length; i++) {
    if (buf[offset + 8 * 4 + i] !== valBytes[i]) return false;
  }
  return true;
});
var absoluteUrl = (relativePath) => typeof document === "undefined" ? new URL(relativePath, self.location.href).href : new URL(relativePath, document.baseURI).href;
var padDigits = (number, digits) => {
  return Array(Math.max(digits - String(number).length + 1, 0)).join("0") + number;
};
var sumArr = (arr) => arr.reduce((prev, curr) => prev + curr, 0);
var isString = (value) => !!(value == null ? void 0 : value.startsWith);
var MMPROJ_FILE_NAME = "mmproj.gguf";
var prepareBlobs = (blobsInp) => __async(void 0, null, function* () {
  const blobs = [];
  let blobMmproj = null;
  for (const blob of blobsInp) {
    if (yield isMmproj(blob)) {
      blobMmproj = blob;
    } else {
      blobs.push(blob);
    }
  }
  sortFileByShard(blobs);
  const result = blobs.map((blob, i) => ({
    blob,
    name: `model-${padDigits(i + 1, 5)}-of-${padDigits(blobs.length, 5)}.gguf`
  }));
  if (blobMmproj) {
    result.push({
      blob: blobMmproj,
      name: MMPROJ_FILE_NAME
    });
  }
  return {
    llm: result.filter((f) => f.name !== MMPROJ_FILE_NAME),
    mmproj: blobMmproj ? { blob: blobMmproj, name: MMPROJ_FILE_NAME } : null,
    all: result
  };
});
var isSupportMultiThread = () => ((e) => __async(void 0, null, function* () {
  try {
    return "undefined" != typeof MessageChannel && new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)), WebAssembly.validate(e);
  } catch (e2) {
    return false;
  }
}))(
  new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    1,
    4,
    1,
    96,
    0,
    0,
    3,
    2,
    1,
    0,
    5,
    4,
    1,
    3,
    1,
    1,
    10,
    11,
    1,
    9,
    0,
    65,
    0,
    254,
    16,
    2,
    0,
    26,
    11
  ])
);
var isSupportExceptions = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      4,
      1,
      96,
      0,
      0,
      3,
      2,
      1,
      0,
      10,
      8,
      1,
      6,
      0,
      6,
      64,
      25,
      11,
      11
    ])
  );
});
var isSupportSIMD = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      5,
      1,
      96,
      0,
      1,
      123,
      3,
      2,
      1,
      0,
      10,
      10,
      1,
      8,
      0,
      65,
      0,
      253,
      15,
      253,
      98,
      11
    ])
  );
});
var isSupportJSPI = () => {
  return !!WebAssembly.Suspending;
};
var isSupportWebGPU = () => {
  return !!navigator.gpu;
};
var isSupportMem64 = () => {
  try {
    new WebAssembly.Memory({
      address: "i64",
      initial: /* @__PURE__ */ BigInt("1")
      // 1 page (64 KiB)
    });
    return true;
  } catch (e) {
    return false;
  }
};
var checkEnvironmentCompatible = () => __async(void 0, null, function* () {
  if (!(yield isSupportExceptions())) {
    throw new Error("WebAssembly runtime does not support exception handling");
  }
  if (!(yield isSupportSIMD())) {
    throw new Error("WebAssembly runtime does not support SIMD");
  }
});
var isFirefox = () => {
  return !!navigator.userAgent.match(/Firefox\/([0-9\.]+)(?:\s|$)/);
};
var GGUF_FILE_REGEX = /^.*\.gguf(?:\?.*)?$/;
var isValidGgufFile = (path) => {
  return GGUF_FILE_REGEX.test(path);
};
var isSafariMobile = () => {
  return !!navigator.userAgent.match(/Version\/([0-9\._]+).*Mobile.*Safari.*/);
};
var createWorker = (workerCode) => {
  const workerURL = URL.createObjectURL(
    isString(workerCode) ? new Blob([workerCode], { type: "text/javascript" }) : workerCode
  );
  return new Worker(workerURL, { type: "module" });
};
var cbToAsyncIter = (fn) => (...args) => {
  let values = [];
  let resolve;
  let reject;
  values.push(
    new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    })
  );
  fn(...args, (val, done, err) => {
    if (err) {
      reject(err);
      return;
    }
    resolve([val, done]);
    values.push(
      new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      })
    );
  });
  return function() {
    return __asyncGenerator(this, null, function* () {
      let val;
      for (let i = 0, done = false; !done; i++) {
        [val, done] = yield new __await(values[i]);
        delete values[i];
        if (val !== void 0) yield val;
      }
    });
  }();
};
var canUseAsyncFileRead = (compat) => isSupportJSPI() || compat;
var needCompat = () => !isSupportJSPI() || !isSupportMem64();

// src/workers-code/generated.ts
var LIBLLAMA_VERSION = "b12096-e8552cc";
var LLAMA_CPP_WORKER_CODE = "// Start the main llama.cpp\nlet wllamaMalloc;\nlet wllamaStart;\nlet wllamaAction;\nlet wllamaExit;\nlet wllamaDebug;\n\nlet Module = null;\nlet isCompat = false;\nlet lastStack = '';\nlet isAborted = false;\nlet hasMultithread = false;\n\n//////////////////////////////////////////////////////////////\n// UTILS\n//////////////////////////////////////////////////////////////\n\n// send message back to main thread\nconst msg = (data, transfer) => postMessage(data, transfer);\n\n// Convert CPP log into JS log\nconst cppLogToJSLog = (line) => {\n  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);\n  return !!matched\n    ? {\n        level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),\n        text: matched[2],\n      }\n    : { level: 'log', text: line };\n};\n\nconst getHeapU8 = () => {\n  const buffer = Module.wasmMemory.buffer;\n  return new Uint8Array(buffer);\n};\n\nconst toSizeT = (num) => {\n  return isCompat ? Number(num) : BigInt(num);\n};\n\n// Get module config that forwards stdout/err to main thread\nconst getWModuleConfig = (_argMainScriptBlob) => {\n  var pathConfig = RUN_OPTIONS.pathConfig;\n  var pthreadPoolSize = RUN_OPTIONS.nbThread;\n  var argMainScriptBlob = _argMainScriptBlob;\n\n  isCompat = RUN_OPTIONS.compat;\n  hasMultithread = pthreadPoolSize > 1;\n\n  msg({\n    verb: 'console.debug',\n    args: [\n      `Multithread enabled: ${hasMultithread}, pthreadPoolSize: ${pthreadPoolSize}`,\n    ],\n  });\n\n  if (!pathConfig['wllama.wasm']) {\n    throw new Error('\"wllama.wasm\" is missing in pathConfig');\n  }\n  return {\n    noInitialRun: true,\n    print: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      msg({ verb: 'console.log', args: [text] });\n    },\n    printErr: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      if (text.startsWith('@@STACK@@')) {\n        lastStack = text.slice('@@STACK@@'.length);\n        return;\n      }\n      const logLine = cppLogToJSLog(text);\n      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });\n    },\n    locateFile: function (filename, basePath) {\n      const p = pathConfig[filename];\n      const truncate = (str) =>\n        str.length > 128 ? `${str.substr(0, 128)}...` : str;\n      if (filename.match(/wllama\\.worker\\.js/)) {\n        msg({\n          verb: 'console.error',\n          args: [\n            '\"wllama.worker.js\" is removed from v2.2.1. Hint: make sure to clear browser\\'s cache.',\n          ],\n        });\n      } else {\n        msg({\n          verb: 'console.debug',\n          args: [`Loading \"${filename}\" from \"${truncate(p)}\"`],\n        });\n        return p;\n      }\n    },\n    mainScriptUrlOrBlob: hasMultithread\n      ? argMainScriptBlob\n      : 'throw new Error(\"Multithreading is not enabled\")',\n    pthreadPoolSize: hasMultithread ? pthreadPoolSize : 0,\n    wasmMemory: hasMultithread ? getWasmMemory() : null,\n    onAbort: function (message) {\n      isAborted = true;\n      msg({ verb: 'signal.abort', args: ['abort', message, lastStack, null] });\n    },\n    onExit: function (code) {\n      isAborted = true;\n      const callstack = new Error().stack.toString();\n      msg({\n        verb: 'signal.abort',\n        args: ['abort', 'exit(' + code + ')', callstack, null],\n      });\n    },\n  };\n};\n\n// Get the memory to be used by wasm. (Only used in multi-thread mode)\n// Because we have a weird OOM issue on iOS, we need to try some values\n// See: https://github.com/emscripten-core/emscripten/issues/19144\n//      https://github.com/godotengine/godot/issues/70621\nconst getWasmMemory = () => {\n  let minBytes = 128 * 1024 * 1024;\n  let maxBytes = 4096 * 1024 * 1024;\n  let stepBytes = 128 * 1024 * 1024;\n  while (maxBytes > minBytes) {\n    try {\n      const wasmMemory = new WebAssembly.Memory({\n        initial: toSizeT(minBytes / 65536),\n        maximum: toSizeT(maxBytes / 65536),\n        shared: true,\n        address: isCompat ? undefined : 'i64',\n      });\n      return wasmMemory;\n    } catch (e) {\n      maxBytes -= stepBytes;\n      continue; // retry\n    }\n  }\n  throw new Error('Cannot allocate WebAssembly.Memory');\n};\n\n//////////////////////////////////////////////////////////////\n// HEAPFS PATCH\n//////////////////////////////////////////////////////////////\n\n/**\n * By default, emscripten uses memfs. The way it works is by\n * allocating new Uint8Array in javascript heap. This is not good\n * because it requires files to be copied to wasm heap each time\n * a file is read.\n *\n * HeapFS is an alternative, which resolves this problem by\n * allocating space for file directly inside wasm heap. This\n * allows us to mmap without doing any copy.\n *\n * For llama.cpp, this is great because we use MAP_SHARED\n *\n * Ref: https://github.com/ngxson/wllama/pull/39\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js\n *\n * Note 29/05/2024 @ngxson\n * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c\n */\n\nconst fsNameToFile = {}; // map Name => File\nconst fsIdToFile = {}; // map ID => File\nlet currFileId = 0;\n\n// Patch and redirect memfs calls to wllama\nconst patchHeapFS = () => {\n  const m = Module;\n  // save functions\n  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;\n  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;\n  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;\n  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;\n  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;\n  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;\n\n  const patchStream = (stream) => {\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const ptr = Number(f.ptr);\n      stream.node.contents = getHeapU8().subarray(ptr, ptr + f.size);\n      stream.node.usedBytes = f.size;\n    }\n  };\n\n  // replace \"read\" functions\n  m.MEMFS.stream_ops.read = function (\n    stream,\n    buffer,\n    offset,\n    length,\n    position\n  ) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);\n  };\n  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;\n\n  // replace \"llseek\" functions\n  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._llseek(stream, offset, whence);\n  };\n  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;\n\n  // replace \"mmap\" functions\n  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {\n    patchStream(stream);\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const mmapPtr = f.ptr + toSizeT(position);\n      return {\n        ptr: mmapPtr,\n        allocated: false,\n      };\n    } else {\n      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);\n    }\n  };\n  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;\n\n  // mount FS\n  m.FS.mkdir('/models');\n  m.FS.mount(m.MEMFS, { root: '.' }, '/models');\n  // slot save files (action \"slots\"), moved to and from the main thread as bytes\n  m.FS.mkdir('/slots');\n};\n\n// Allocate a new file in wllama heapfs, returns file ID\nconst heapfsAlloc = (name, size, allocBuffer) => {\n  if (size < 1) {\n    throw new Error('File size must be bigger than 0');\n  }\n  const m = Module;\n  const ptr = toSizeT(allocBuffer ? m.mmapAlloc(size) : 0);\n  const file = {\n    ptr: ptr,\n    size: size,\n    id: currFileId++,\n  };\n  fsIdToFile[file.id] = file;\n  fsNameToFile[name] = file;\n  return file.id;\n};\n\n// Add new file to wllama heapfs, return number of written bytes\nconst heapfsWrite = (id, buffer, offset) => {\n  if (fsIdToFile[id]) {\n    const { ptr, size } = fsIdToFile[id];\n    const afterWriteByte = offset + buffer.byteLength;\n    if (afterWriteByte > size) {\n      throw new Error(\n        `File ID ${id} write out of bound, afterWriteByte = ${afterWriteByte} while size = ${size}`\n      );\n    }\n    getHeapU8().set(buffer, Number(ptr) + offset);\n    return buffer.byteLength;\n  } else {\n    throw new Error(`File ID ${id} not found in heapfs`);\n  }\n};\n\n//////////////////////////////////////////////////////////////\n// ASYNC FILE READ\n//////////////////////////////////////////////////////////////\n\nlet isAwaitReading = false;\nlet pendingReadPromise = null;\nlet pendingReadResolve = null;\nlet pendingReadReject = null;\n\nconst _stripModelsPrefix = (path) => path.replace(/^\\/?models\\//, '');\n\n// Called from EM_ASYNC_JS stub in wllama-fs.h (path is already a JS string)\nconst _wllama_js_file_read = async (path, offset, req_size, out_ptr) => {\n  const name = _stripModelsPrefix(path);\n\n  pendingReadPromise = new Promise((res, rej) => {\n    pendingReadResolve = res;\n    pendingReadReject = rej;\n  });\n  isAwaitReading = true;\n\n  postMessage({ verb: 'fs.read_req', args: [name, offset, req_size] });\n\n  let data;\n  try {\n    data = await pendingReadPromise;\n  } finally {\n    isAwaitReading = false;\n    pendingReadResolve = null;\n    pendingReadReject = null;\n  }\n\n  const bytes = new Uint8Array(data);\n  getHeapU8().set(bytes, out_ptr);\n  return toSizeT(bytes.length);\n};\n\n//////////////////////////////////////////////////////////////\n// MAIN CODE\n//////////////////////////////////////////////////////////////\n\nconst callWrapper = (name, ret, args, isAsync) => {\n  const fn = Module.cwrap(\n    name,\n    ret,\n    args,\n    isAsync ? { async: true } : undefined\n  );\n  return async (action, req) => {\n    // console.log(`Calling ${name} with action:`, action, 'and req:', req);\n    let result;\n    try {\n      if (args.length === 2) {\n        result = isAsync ? await fn(action, req) : fn(action, req);\n      } else {\n        result = fn();\n      }\n    } catch (ex) {\n      console.error(ex);\n      throw ex;\n    }\n    return result;\n  };\n};\n\n// re-entering the wasm while a call is suspended (JSPI / asyncify) corrupts its state, so only one call runs at a time and the rest wait in the queue\nlet wasmCallBusy = false;\nconst wasmCallQueue = [];\n\nconst runWasmCall = async (callbackId, fn) => {\n  if (isAborted) {\n    // the wasm is dead, fail fast instead of calling into it\n    msg({ callbackId, err: 'wllama has crashed, please reload the module' });\n    return;\n  }\n  if (wasmCallBusy) {\n    wasmCallQueue.push({ callbackId, fn });\n    return;\n  }\n  wasmCallBusy = true;\n  try {\n    await fn();\n  } finally {\n    wasmCallBusy = false;\n    if (isAborted) {\n      // do not touch the wasm again after it aborted; the main thread already rejected the queued tasks\n      wasmCallQueue.length = 0;\n    } else {\n      const next = wasmCallQueue.shift();\n      if (next) runWasmCall(next.callbackId, next.fn);\n    }\n  }\n};\n\nconst runAction = async (data) => {\n  const { args, callbackId } = data;\n  const argAction = args[0];\n  const argEncodedMsg = args[1];\n  try {\n    const inputPtr = await wllamaMalloc(toSizeT(argEncodedMsg.byteLength), 0);\n    // copy data to wasm heap\n    const inputBuffer = new Uint8Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      argEncodedMsg.byteLength\n    );\n    inputBuffer.set(argEncodedMsg, 0);\n    const outputPtr = await wllamaAction(argAction, inputPtr);\n    // length of output buffer is written at the first 4 bytes of input buffer\n    const outputLen = new Uint32Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      1\n    )[0];\n    // copy the output buffer to JS heap\n    const outputBuffer = new Uint8Array(outputLen);\n    const outputSrcView = new Uint8Array(\n      getHeapU8().buffer,\n      Number(outputPtr),\n      outputLen\n    );\n    outputBuffer.set(outputSrcView, 0); // copy it\n    msg({ callbackId, result: outputBuffer }, [outputBuffer.buffer]);\n  } catch (err) {\n    handleError(err);\n  }\n};\n\nfunction handleError(err) {\n  // If WASM already aborted, onAbort already sent signal.abort; skip to avoid\n  // re-reporting the resulting WebAssembly.RuntimeError as a JS exception.\n  if (isAborted) return;\n\n  const message = err ? err.message || String(err) : 'Unknown error';\n  const stack = err ? err.stack || String(err) : '';\n  msg({\n    verb: 'signal.abort',\n    args: ['exception', message, stack, err],\n  });\n}\n\nonmessage = async (e) => {\n  if (!e.data) return;\n  const { verb, args, callbackId } = e.data;\n\n  // fs.read_res arrives while wasm is JSPI-suspended; resolve the pending promise.\n  if (verb === 'fs.read_res') {\n    if (pendingReadResolve) {\n      pendingReadResolve(args[0]);\n    }\n    return;\n  }\n\n  // Guard: while awaiting a file read, reject any other incoming task.\n  if (isAwaitReading) {\n    if (callbackId) {\n      msg({\n        callbackId,\n        err: 'Worker is suspended waiting for file data (JSPI)',\n      });\n    }\n    return;\n  }\n\n  if (!callbackId) {\n    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });\n    return;\n  }\n\n  if (verb === 'module.init') {\n    const argMainScriptBlob = args[0];\n    const argUseAsyncFile = args[1];\n    try {\n      Module = getWModuleConfig(argMainScriptBlob);\n      Module.preRun = () => {\n        if (argUseAsyncFile) {\n          Module.ENV['USE_ASYNC_FILE'] = '1';\n        }\n      };\n      Module.onRuntimeInitialized = () => {\n        // async call once module is ready\n        // init FS\n        patchHeapFS();\n        // init cwrap\n        const pointer = isCompat ? 'number' : 'bigint';\n        // TODO: note sure why emscripten cannot bind if there is only 1 argument\n        wllamaMalloc = callWrapper('wllama_malloc', pointer, [\n          'number',\n          pointer,\n        ]);\n        wllamaStart = callWrapper('wllama_start', 'string', [], true);\n        wllamaAction = callWrapper(\n          'wllama_action',\n          pointer,\n          ['string', pointer],\n          true\n        );\n        wllamaExit = callWrapper('wllama_exit', 'string', []);\n        wllamaDebug = callWrapper('wllama_debug', 'string', []);\n        msg({ callbackId, result: null });\n      };\n      wModuleInit();\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.alloc') {\n    const argFilename = args[0];\n    const argSize = args[1];\n    const argAllocBuffer = args[2];\n    try {\n      // create blank file\n      const emptyBuffer = new ArrayBuffer(0);\n      Module['FS_createDataFile'](\n        '/models',\n        argFilename,\n        emptyBuffer,\n        true,\n        true,\n        true\n      );\n      // alloc data on heap\n      const fileId = heapfsAlloc(argFilename, argSize, argAllocBuffer);\n      msg({ callbackId, result: { fileId } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.write') {\n    const argFileId = args[0];\n    const argBuffer = args[1];\n    const argOffset = args[2];\n    try {\n      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);\n      msg({ callbackId, result: { writtenBytes } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_read') {\n    const argFilename = args[0];\n    try {\n      const path = '/slots/' + argFilename;\n      const data = Module.FS.readFile(path);\n      Module.FS.unlink(path);\n      msg({ callbackId, result: data }, [data.buffer]);\n    } catch (err) {\n      msg({ callbackId, err: `cannot read slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_write') {\n    const argFilename = args[0];\n    const argData = args[1];\n    try {\n      Module.FS.writeFile('/slots/' + argFilename, argData);\n      msg({ callbackId, result: null });\n    } catch (err) {\n      msg({ callbackId, err: `cannot write slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_remove') {\n    try {\n      Module.FS.unlink('/slots/' + args[0]);\n    } catch (err) {\n      // already removed\n    }\n    msg({ callbackId, result: null });\n    return;\n  }\n\n  if (verb === 'wllama.start') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaStart();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.action') {\n    await runWasmCall(callbackId, () => runAction(e.data));\n    return;\n  }\n\n  if (verb === 'wllama.exit') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaExit();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.debug') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaDebug();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n};\n";
var OPFS_UTILS_WORKER_CODE = "let accessHandle;\nlet abortController = new AbortController();\n\nasync function openFile(filename) {\n  const opfsRoot = await navigator.storage.getDirectory();\n  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });\n  const fileHandler = await cacheDir.getFileHandle(filename, { create: true });\n  accessHandle = await fileHandler.createSyncAccessHandle();\n  accessHandle.truncate(0); // clear file content\n}\n\nasync function writeFile(buf) {\n  accessHandle.write(buf);\n}\n\nasync function closeFile() {\n  accessHandle.flush();\n  accessHandle.close();\n}\n\nasync function writeTextFile(filename, str) {\n  await openFile(filename);\n  await writeFile(new TextEncoder().encode(str));\n  await closeFile();\n}\n\nconst throttled = (func, delay) => {\n  let lastRun = 0;\n  return (...args) => {\n    const now = Date.now();\n    if (now - lastRun > delay) {\n      lastRun = now;\n      func.apply(null, args);\n    }\n  };\n};\n\nconst assertNonNull = (val) => {\n  if (val === null || val === undefined) {\n    throw new Error('OPFS Worker: Assertion failed');\n  }\n};\n\n// respond to main thread\nconst resOK = () => postMessage({ ok: true });\nconst resProgress = (loaded, total) =>\n  postMessage({ progress: { loaded, total } });\nconst resErr = (err) => postMessage({ err });\n\nonmessage = async (e) => {\n  try {\n    if (!e.data) return;\n\n    /**\n     * @param {Object} e.data\n     *\n     * Fine-control FS actions:\n     * - { action: 'open', filename: 'string' }\n     * - { action: 'write', buf: ArrayBuffer }\n     * - { action: 'close' }\n     *\n     * Simple write API:\n     * - { action: 'write-simple', filename: 'string', buf: ArrayBuffer }\n     *\n     * Download API:\n     * - { action: 'download', url: 'string', filename: 'string', options: Object, metadataFileName: 'string' }\n     * - { action: 'download-abort' }\n     */\n    const {\n      action,\n      filename,\n      buf,\n      url,\n      options,\n      metadataFileName,\n      metadataAdditional,\n    } = e.data;\n\n    if (action === 'open') {\n      assertNonNull(filename);\n      await openFile(filename);\n      return resOK();\n    } else if (action === 'write') {\n      assertNonNull(buf);\n      await writeFile(buf);\n      return resOK();\n    } else if (action === 'close') {\n      await closeFile();\n      return resOK();\n    } else if (action === 'write-simple') {\n      assertNonNull(filename);\n      assertNonNull(buf);\n      await openFile(filename);\n      await writeFile(buf);\n      await closeFile();\n      return resOK();\n    } else if (action === 'download') {\n      assertNonNull(url);\n      assertNonNull(filename);\n      assertNonNull(metadataFileName);\n      assertNonNull(options);\n      assertNonNull(options.aborted);\n      abortController = new AbortController();\n      if (options.aborted) abortController.abort();\n      const response = await fetch(url, {\n        ...options,\n        signal: abortController.signal,\n      });\n      const contentLength = response.headers.get('content-length');\n      const etag = (response.headers.get('etag') || '').replace(\n        /[^A-Za-z0-9]/g,\n        ''\n      );\n      const total = parseInt(contentLength, 10);\n      const reader = response.body.getReader();\n      await openFile(filename);\n      let loaded = 0;\n      const throttledProgress = throttled(resProgress, 100);\n      while (true) {\n        const { done, value } = await reader.read();\n        if (done) break;\n        loaded += value.byteLength;\n        await writeFile(value);\n        throttledProgress(loaded, total);\n      }\n      resProgress(total, total); // 100% done\n      await closeFile();\n      // make sure this is in-sync with CacheEntryMetadata\n      await writeTextFile(\n        metadataFileName,\n        JSON.stringify({\n          originalURL: url,\n          originalSize: total,\n          etag,\n          ...metadataAdditional,\n        })\n      );\n      return resOK();\n    } else if (action === 'download-abort') {\n      if (abortController) {\n        abortController.abort();\n      }\n      return;\n    }\n\n    throw new Error('OPFS Worker: Invalid action', e.data);\n  } catch (err) {\n    return resErr(err);\n  }\n};\n";
var WLLAMA_EMSCRIPTEN_CODE = 'var Module=typeof Module!="undefined"?Module:{};var ENVIRONMENT_IS_WEB=!!globalThis.window;var ENVIRONMENT_IS_WORKER=!!globalThis.WorkerGlobalScope;var ENVIRONMENT_IS_NODE=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";var ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&self.name?.startsWith("em-pthread");if(ENVIRONMENT_IS_NODE){var worker_threads=require("worker_threads");global.Worker=worker_threads.Worker;ENVIRONMENT_IS_WORKER=!worker_threads.isMainThread;ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&worker_threads["workerData"]=="em-pthread"}var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var _scriptName=globalThis.document?.currentScript?.src;if(typeof __filename!="undefined"){_scriptName=__filename}else if(ENVIRONMENT_IS_WORKER){_scriptName=self.location.href}var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("fs");scriptDirectory=__dirname+"/";readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename);return ret};readAsync=async(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename,binary?undefined:"utf8");return ret};if(process.argv.length>1){thisProgram=process.argv[1].replace(/\\\\/g,"/")}arguments_=process.argv.slice(2);if(typeof module!="undefined"){module["exports"]=Module}quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){try{scriptDirectory=new URL(".",_scriptName).href}catch{}if(!ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=async url=>{if(isFileURI(url)){return new Promise((resolve,reject)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){resolve(xhr.response);return}reject(xhr.status)};xhr.onerror=reject;xhr.send(null)})}var response=await fetch(url,{credentials:"same-origin"});if(response.ok){return response.arrayBuffer()}throw new Error(response.status+" : "+response.url)}}}else{}var defaultPrint=console.log.bind(console);var defaultPrintErr=console.error.bind(console);if(ENVIRONMENT_IS_NODE){var utils=require("util");var stringify=a=>typeof a=="object"?utils.inspect(a):a;defaultPrint=(...args)=>fs.writeSync(1,args.map(stringify).join(" ")+"\\n");defaultPrintErr=(...args)=>fs.writeSync(2,args.map(stringify).join(" ")+"\\n")}var out=defaultPrint;var err=defaultPrintErr;var wasmBinary;var wasmModule;var ABORT=false;var EXITSTATUS;function assert(condition,text){if(!condition){abort(text)}}var isFileURI=filename=>filename.startsWith("file://");function growMemViews(){if(wasmMemory.buffer!=HEAP8.buffer){updateMemoryViews()}}if(ENVIRONMENT_IS_NODE&&ENVIRONMENT_IS_PTHREAD){var parentPort=worker_threads["parentPort"];parentPort.on("message",msg=>global.onmessage?.({data:msg}));Object.assign(globalThis,{self:global,postMessage:msg=>parentPort["postMessage"](msg)});process.on("uncaughtException",err=>{postMessage({cmd:"uncaughtException",error:err});process.exit(1)})}var startWorker;if(ENVIRONMENT_IS_PTHREAD){var initializedJS=false;self.onunhandledrejection=e=>{throw e.reason||e};async function handleMessage(e){try{var msgData=e["data"];var cmd=msgData.cmd;if(cmd==="load"){let messageQueue=[];self.onmessage=e=>messageQueue.push(e);startWorker=()=>{postMessage({cmd:"loaded"});for(let msg of messageQueue){handleMessage(msg)}self.onmessage=handleMessage};for(const handler of msgData.handlers){if(!Module[handler]||Module[handler].proxy){Module[handler]=(...args)=>{postMessage({cmd:"callHandler",handler,args})};if(handler=="print")out=Module[handler];if(handler=="printErr")err=Module[handler]}}wasmMemory=msgData.wasmMemory;updateMemoryViews();wasmModule=msgData.wasmModule;createWasm();run()}else if(cmd==="run"){establishStackSpace(msgData.pthread_ptr);__emscripten_thread_init(msgData.pthread_ptr,0,0,1,0,0);PThread.threadInitTLS();__emscripten_thread_mailbox_await(msgData.pthread_ptr);if(!initializedJS){initializedJS=true}try{await invokeEntryPoint(msgData.start_routine,msgData.arg)}catch(ex){if(ex!="unwind"){throw ex}}}else if(msgData.target==="setimmediate"){}else if(cmd==="checkMailbox"){if(initializedJS){checkMailbox()}}else if(cmd){err(`worker: received unknown command ${cmd}`);err(msgData)}}catch(ex){__emscripten_thread_crashed();throw ex}}self.onmessage=handleMessage}var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;var HEAP64,HEAPU64;var runtimeInitialized=false;function updateMemoryViews(){var b=wasmMemory.buffer;HEAP8=new Int8Array(b);HEAP16=new Int16Array(b);Module["HEAPU8"]=HEAPU8=new Uint8Array(b);HEAPU16=new Uint16Array(b);HEAP32=new Int32Array(b);HEAPU32=new Uint32Array(b);HEAPF32=new Float32Array(b);HEAPF64=new Float64Array(b);HEAP64=new BigInt64Array(b);HEAPU64=new BigUint64Array(b)}function initMemory(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"]}else{var INITIAL_MEMORY=Module["INITIAL_MEMORY"]||134217728;wasmMemory=new WebAssembly.Memory({initial:BigInt(INITIAL_MEMORY/65536),maximum:65536n,shared:true,address:"i64"})}updateMemoryViews()}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift())}}callRuntimeCallbacks(onPreRuns)}function initRuntime(){runtimeInitialized=true;if(ENVIRONMENT_IS_PTHREAD)return startWorker();if(!Module["noFSInit"]&&!FS.initialized)FS.init();TTY.init();wasmExports["__wasm_call_ctors"]();FS.ignorePermissions=false}function preMain(){}function postRun(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift())}}callRuntimeCallbacks(onPostRuns)}function abort(what){Module["onAbort"]?.(what);what="Aborted("+what+")";err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";if(runtimeInitialized){___trap()}var e=new WebAssembly.RuntimeError(what);throw e}var wasmBinaryFile;function findWasmBinary(){return locateFile("wllama.wasm")}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}async function getWasmBinary(binaryFile){if(!wasmBinary){try{var response=await readAsync(binaryFile);return new Uint8Array(response)}catch{}}return getBinarySync(binaryFile)}async function instantiateArrayBuffer(binaryFile,imports){try{var binary=await getWasmBinary(binaryFile);var instance=await WebAssembly.instantiate(binary,imports);return instance}catch(reason){err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)}}async function instantiateAsync(binary,binaryFile,imports){if(!binary&&!isFileURI(binaryFile)&&!ENVIRONMENT_IS_NODE){try{var response=fetch(binaryFile,{credentials:"same-origin"});var instantiationResult=await WebAssembly.instantiateStreaming(response,imports);return instantiationResult}catch(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation")}}return instantiateArrayBuffer(binaryFile,imports)}function getWasmImports(){assignWasmImports();if(!wasmImports.__instrumented){wasmImports.__instrumented=true;Asyncify.instrumentWasmImports(wasmImports)}var imports={env:wasmImports,wasi_snapshot_preview1:wasmImports};return imports}async function createWasm(){function receiveInstance(instance,module){wasmExports=instance.exports;wasmExports=Asyncify.instrumentWasmExports(wasmExports);wasmExports=applySignatureConversions(wasmExports);registerTLSInit(wasmExports["_emscripten_tls_init"]);assignWasmExports(wasmExports);wasmModule=module;removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){return receiveInstance(result["instance"],result["module"])}var info=getWasmImports();if(Module["instantiateWasm"]){return new Promise((resolve,reject)=>{Module["instantiateWasm"](info,(inst,mod)=>{resolve(receiveInstance(inst,mod))})})}if(ENVIRONMENT_IS_PTHREAD){var instance=new WebAssembly.Instance(wasmModule,getWasmImports());return receiveInstance(instance,wasmModule)}wasmBinaryFile??=findWasmBinary();var result=await instantiateAsync(wasmBinary,wasmBinaryFile,info);var exports=receiveInstantiationResult(result);return exports}class ExitStatus{name="ExitStatus";constructor(status){this.message=`Program terminated with exit(${status})`;this.status=status}}var terminateWorker=worker=>{worker.terminate();worker.onmessage=e=>{}};var cleanupThread=pthread_ptr=>{var worker=PThread.pthreads[pthread_ptr];PThread.returnWorkerToPool(worker)};var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module)}};var onPreRuns=[];var addOnPreRun=cb=>onPreRuns.push(cb);var runDependencies=0;var dependenciesFulfilled=null;var removeRunDependency=id=>{runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}};var addRunDependency=id=>{runDependencies++;Module["monitorRunDependencies"]?.(runDependencies)};var spawnThread=threadParams=>{var worker=PThread.getNewWorker();if(!worker){return 6}PThread.runningWorkers.push(worker);PThread.pthreads[threadParams.pthread_ptr]=worker;worker.pthread_ptr=threadParams.pthread_ptr;var msg={cmd:"run",start_routine:threadParams.startRoutine,arg:threadParams.arg,pthread_ptr:threadParams.pthread_ptr};if(ENVIRONMENT_IS_NODE){worker.unref()}worker.postMessage(msg,threadParams.transferList);return 0};var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var stackSave=()=>_emscripten_stack_get_current();var stackRestore=val=>__emscripten_stack_restore(val);var stackAlloc=sz=>__emscripten_stack_alloc(sz);var proxyToMainThread=(funcIndex,emAsmAddr,sync,...callArgs)=>{var serializedNumCallArgs=callArgs.length*2;var sp=stackSave();var args=stackAlloc(serializedNumCallArgs*8);var b=args/8;for(var i=0;i<callArgs.length;i++){var arg=callArgs[i];if(typeof arg=="bigint"){(growMemViews(),HEAP64)[b+2*i]=1n;(growMemViews(),HEAP64)[b+2*i+1]=arg}else{(growMemViews(),HEAP64)[b+2*i]=0n;(growMemViews(),HEAPF64)[b+2*i+1]=arg}}var rtn=__emscripten_run_js_on_main_thread(funcIndex,emAsmAddr,serializedNumCallArgs,args,sync);stackRestore(sp);return rtn};function _proc_exit(code){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(0,0,1,code);EXITSTATUS=code;if(!keepRuntimeAlive()){PThread.terminateAllThreads();Module["onExit"]?.(code);ABORT=true}quit_(code,new ExitStatus(code))}function exitOnMainThread(returnCode){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(1,0,0,returnCode);_exit(returnCode)}var exitJS=(status,implicit)=>{EXITSTATUS=status;if(ENVIRONMENT_IS_PTHREAD){exitOnMainThread(status);throw"unwind"}_proc_exit(status)};var _exit=exitJS;var PThread={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init(){if(!ENVIRONMENT_IS_PTHREAD){PThread.initMainThread()}},initMainThread(){var pthreadPoolSize=Module["pthreadPoolSize"];while(pthreadPoolSize--){PThread.allocateUnusedWorker()}addOnPreRun(async()=>{var pthreadPoolReady=PThread.loadWasmModuleToAllWorkers();addRunDependency("loading-workers");await pthreadPoolReady;removeRunDependency("loading-workers")})},terminateAllThreads:()=>{for(var worker of PThread.runningWorkers){terminateWorker(worker)}for(var worker of PThread.unusedWorkers){terminateWorker(worker)}PThread.unusedWorkers=[];PThread.runningWorkers=[];PThread.pthreads={}},returnWorkerToPool:worker=>{var pthread_ptr=worker.pthread_ptr;delete PThread.pthreads[pthread_ptr];PThread.unusedWorkers.push(worker);PThread.runningWorkers.splice(PThread.runningWorkers.indexOf(worker),1);worker.pthread_ptr=0;__emscripten_thread_free_data(pthread_ptr)},threadInitTLS(){PThread.tlsInitFunctions.forEach(f=>f())},loadWasmModuleToWorker:worker=>new Promise(onFinishedLoading=>{worker.onmessage=e=>{var d=e["data"];var cmd=d.cmd;if(d.targetThread&&d.targetThread!=_pthread_self()){var targetWorker=PThread.pthreads[d.targetThread];if(targetWorker){targetWorker.postMessage(d,d.transferList)}else{err(`Internal error! Worker sent a message "${cmd}" to target pthread ${d.targetThread}, but that thread no longer exists!`)}return}if(cmd==="checkMailbox"){checkMailbox()}else if(cmd==="spawnThread"){spawnThread(d)}else if(cmd==="cleanupThread"){callUserCallback(()=>cleanupThread(d.thread))}else if(cmd==="loaded"){worker.loaded=true;if(ENVIRONMENT_IS_NODE&&!worker.pthread_ptr){worker.unref()}onFinishedLoading(worker)}else if(d.target==="setimmediate"){worker.postMessage(d)}else if(cmd==="uncaughtException"){worker.onerror(d.error)}else if(cmd==="callHandler"){Module[d.handler](...d.args)}else if(cmd){err(`worker sent an unknown command ${cmd}`)}};worker.onerror=e=>{var message="worker sent an error!";err(`${message} ${e.filename}:${e.lineno}: ${e.message}`);throw e};if(ENVIRONMENT_IS_NODE){worker.on("message",data=>worker.onmessage({data}));worker.on("error",e=>worker.onerror(e))}var handlers=[];var knownHandlers=["onExit","onAbort","print","printErr"];for(var handler of knownHandlers){if(Module.propertyIsEnumerable(handler)){handlers.push(handler)}}worker.postMessage({cmd:"load",handlers,wasmMemory,wasmModule})}),async loadWasmModuleToAllWorkers(){if(ENVIRONMENT_IS_PTHREAD){return}let pthreadPoolReady=Promise.all(PThread.unusedWorkers.map(PThread.loadWasmModuleToWorker));return pthreadPoolReady},allocateUnusedWorker(){var worker;var pthreadMainJs=_scriptName;if(Module["mainScriptUrlOrBlob"]){pthreadMainJs=Module["mainScriptUrlOrBlob"];if(typeof pthreadMainJs!="string"){pthreadMainJs=URL.createObjectURL(pthreadMainJs)}}worker=new Worker(pthreadMainJs,{workerData:"em-pthread",name:"em-pthread"});PThread.unusedWorkers.push(worker)},getNewWorker(){if(PThread.unusedWorkers.length==0){PThread.allocateUnusedWorker();PThread.loadWasmModuleToWorker(PThread.unusedWorkers[0])}return PThread.unusedWorkers.pop()}};var onPostRuns=[];var addOnPostRun=cb=>onPostRuns.push(cb);function establishStackSpace(pthread_ptr){var stackHigh=Number((growMemViews(),HEAPU64)[(pthread_ptr+88)/8]);var stackSize=Number((growMemViews(),HEAPU64)[(pthread_ptr+96)/8]);var stackLow=stackHigh-stackSize;_emscripten_stack_set_limits(stackHigh,stackLow);stackRestore(stackHigh)}var wasmTableMirror=[];var getWasmTableEntry=funcPtr=>{funcPtr=Number(funcPtr);var func=wasmTableMirror[funcPtr];if(!func){wasmTableMirror[funcPtr]=func=wasmTable.get(BigInt(funcPtr));if(Asyncify.isAsyncExport(func)){wasmTableMirror[funcPtr]=func=Asyncify.makeAsyncFunction(func)}}return func};var invokeEntryPoint=async(ptr,arg)=>{runtimeKeepaliveCounter=0;noExitRuntime=0;var result=(a1=>WebAssembly.promising(getWasmTableEntry(ptr)).call(null,BigInt(a1)))(arg);function finish(result){if(keepRuntimeAlive()){EXITSTATUS=result;return}__emscripten_thread_exit(result)}result=await result;finish(result)};invokeEntryPoint.isAsync=true;var noExitRuntime=true;var registerTLSInit=tlsInitFunc=>PThread.tlsInitFunctions.push(tlsInitFunc);var wasmMemory;function pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(2,0,1,pthread_ptr,attr,startRoutine,arg);return ___pthread_create_js(pthread_ptr,attr,startRoutine,arg)}var _emscripten_has_threading_support=()=>!!globalThis.SharedArrayBuffer;var INT53_MAX=9007199254740992;var INT53_MIN=-9007199254740992;var bigintToI53Checked=num=>num<INT53_MIN||num>INT53_MAX?NaN:Number(num);function ___pthread_create_js(pthread_ptr,attr,startRoutine,arg){pthread_ptr=bigintToI53Checked(pthread_ptr);attr=bigintToI53Checked(attr);startRoutine=bigintToI53Checked(startRoutine);arg=bigintToI53Checked(arg);if(!_emscripten_has_threading_support()){return 6}var transferList=[];var error=0;if(ENVIRONMENT_IS_PTHREAD&&(transferList.length===0||error)){return pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg)}if(error)return error;var threadParams={startRoutine,pthread_ptr,arg,transferList};if(ENVIRONMENT_IS_PTHREAD){threadParams.cmd="spawnThread";postMessage(threadParams,transferList);return 0}return spawnThread(threadParams)}var syscallGetVarargP=()=>{var ret=Number((growMemViews(),HEAPU64)[SYSCALLS.varargs/8]);SYSCALLS.varargs+=8;return ret};var syscallGetVarargI=()=>{var ret=(growMemViews(),HEAP32)[+SYSCALLS.varargs/4];SYSCALLS.varargs+=4;return ret};var PATH={isAbs:path=>path.charAt(0)==="/",splitPath:filename=>{var splitPathRe=/^(\\/?|)([\\s\\S]*?)((?:\\.{1,2}|[^\\/]+?|)(\\.[^.\\/]*|))(?:[\\/]*)$/;return splitPathRe.exec(filename).slice(1)},normalizeArray:(parts,allowAboveRoot)=>{var up=0;for(var i=parts.length-1;i>=0;i--){var last=parts[i];if(last==="."){parts.splice(i,1)}else if(last===".."){parts.splice(i,1);up++}else if(up){parts.splice(i,1);up--}}if(allowAboveRoot){for(;up;up--){parts.unshift("..")}}return parts},normalize:path=>{var isAbsolute=PATH.isAbs(path),trailingSlash=path.slice(-1)==="/";path=PATH.normalizeArray(path.split("/").filter(p=>!!p),!isAbsolute).join("/");if(!path&&!isAbsolute){path="."}if(path&&trailingSlash){path+="/"}return(isAbsolute?"/":"")+path},dirname:path=>{var result=PATH.splitPath(path),root=result[0],dir=result[1];if(!root&&!dir){return"."}if(dir){dir=dir.slice(0,-1)}return root+dir},basename:path=>path&&path.match(/([^\\/]+|\\/)\\/*$/)[1],join:(...paths)=>PATH.normalize(paths.join("/")),join2:(l,r)=>PATH.normalize(l+"/"+r)};var initRandomFill=()=>view=>view.set(crypto.getRandomValues(new Uint8Array(view.byteLength)));var randomFill=view=>{(randomFill=initRandomFill())(view)};var PATH_FS={resolve:(...args)=>{var resolvedPath="",resolvedAbsolute=false;for(var i=args.length-1;i>=-1&&!resolvedAbsolute;i--){var path=i>=0?args[i]:FS.cwd();if(typeof path!="string"){throw new TypeError("Arguments to path.resolve must be strings")}else if(!path){return""}resolvedPath=path+"/"+resolvedPath;resolvedAbsolute=PATH.isAbs(path)}resolvedPath=PATH.normalizeArray(resolvedPath.split("/").filter(p=>!!p),!resolvedAbsolute).join("/");return(resolvedAbsolute?"/":"")+resolvedPath||"."},relative:(from,to)=>{from=PATH_FS.resolve(from).slice(1);to=PATH_FS.resolve(to).slice(1);function trim(arr){var start=0;for(;start<arr.length;start++){if(arr[start]!=="")break}var end=arr.length-1;for(;end>=0;end--){if(arr[end]!=="")break}if(start>end)return[];return arr.slice(start,end-start+1)}var fromParts=trim(from.split("/"));var toParts=trim(to.split("/"));var length=Math.min(fromParts.length,toParts.length);var samePartsLength=length;for(var i=0;i<length;i++){if(fromParts[i]!==toParts[i]){samePartsLength=i;break}}var outputParts=[];for(var i=samePartsLength;i<fromParts.length;i++){outputParts.push("..")}outputParts=outputParts.concat(toParts.slice(samePartsLength));return outputParts.join("/")}};var UTF8Decoder=globalThis.TextDecoder&&new TextDecoder;var findStringEnd=(heapOrArray,idx,maxBytesToRead,ignoreNul)=>{var maxIdx=idx+maxBytesToRead;if(ignoreNul)return maxIdx;while(heapOrArray[idx]&&!(idx>=maxIdx))++idx;return idx};var UTF8ArrayToString=(heapOrArray,idx=0,maxBytesToRead,ignoreNul)=>{var endPtr=findStringEnd(heapOrArray,idx,maxBytesToRead,ignoreNul);if(endPtr-idx>16&&heapOrArray.buffer&&UTF8Decoder){return UTF8Decoder.decode(heapOrArray.buffer instanceof ArrayBuffer?heapOrArray.subarray(idx,endPtr):heapOrArray.slice(idx,endPtr))}var str="";while(idx<endPtr){var u0=heapOrArray[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heapOrArray[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heapOrArray[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2}else{u0=(u0&7)<<18|u1<<12|u2<<6|heapOrArray[idx++]&63}if(u0<65536){str+=String.fromCharCode(u0)}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023)}}return str};var FS_stdin_getChar_buffer=[];var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++}else if(c<=2047){len+=2}else if(c>=55296&&c<=57343){len+=4;++i}else{len+=3}}return len};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.codePointAt(i);if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;i++}}heap[outIdx]=0;return outIdx-startIdx};var intArrayFromString=(stringy,dontAddNull,length)=>{var len=length>0?length:lengthBytesUTF8(stringy)+1;var u8array=new Array(len);var numBytesWritten=stringToUTF8Array(stringy,u8array,0,u8array.length);if(dontAddNull)u8array.length=numBytesWritten;return u8array};var FS_stdin_getChar=()=>{if(!FS_stdin_getChar_buffer.length){var result=null;if(ENVIRONMENT_IS_NODE){var BUFSIZE=256;var buf=Buffer.alloc(BUFSIZE);var bytesRead=0;var fd=process.stdin.fd;try{bytesRead=fs.readSync(fd,buf,0,BUFSIZE)}catch(e){if(e.toString().includes("EOF"))bytesRead=0;else throw e}if(bytesRead>0){result=buf.slice(0,bytesRead).toString("utf-8")}}else if(globalThis.window?.prompt){result=window.prompt("Input: ");if(result!==null){result+="\\n"}}else{}if(!result){return null}FS_stdin_getChar_buffer=intArrayFromString(result,true)}return FS_stdin_getChar_buffer.shift()};var TTY={ttys:[],init(){},shutdown(){},register(dev,ops){TTY.ttys[dev]={input:[],output:[],ops};FS.registerDevice(dev,TTY.stream_ops)},stream_ops:{open(stream){var tty=TTY.ttys[stream.node.rdev];if(!tty){throw new FS.ErrnoError(43)}stream.tty=tty;stream.seekable=false},close(stream){stream.tty.ops.fsync(stream.tty)},fsync(stream){stream.tty.ops.fsync(stream.tty)},read(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.get_char){throw new FS.ErrnoError(60)}var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=stream.tty.ops.get_char(stream.tty)}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.put_char){throw new FS.ErrnoError(60)}try{for(var i=0;i<length;i++){stream.tty.ops.put_char(stream.tty,buffer[offset+i])}}catch(e){throw new FS.ErrnoError(29)}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}},default_tty_ops:{get_char(tty){return FS_stdin_getChar()},put_char(tty,val){if(val===null||val===10){out(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){out(UTF8ArrayToString(tty.output));tty.output=[]}},ioctl_tcgets(tty){return{c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(tty,optional_actions,data){return 0},ioctl_tiocgwinsz(tty){return[24,80]}},default_tty1_ops:{put_char(tty,val){if(val===null||val===10){err(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){err(UTF8ArrayToString(tty.output));tty.output=[]}}}};var zeroMemory=(ptr,size)=>(growMemViews(),HEAPU8).fill(0,ptr,ptr+size);var alignMemory=(size,alignment)=>Math.ceil(size/alignment)*alignment;var mmapAlloc=size=>{size=alignMemory(size,65536);var ptr=_emscripten_builtin_memalign(65536,size);if(ptr)zeroMemory(ptr,size);return ptr};var MEMFS={ops_table:null,mount(mount){return MEMFS.createNode(null,"/",16895,0)},createNode(parent,name,mode,dev){if(FS.isBlkdev(mode)||FS.isFIFO(mode)){throw new FS.ErrnoError(63)}MEMFS.ops_table||={dir:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,lookup:MEMFS.node_ops.lookup,mknod:MEMFS.node_ops.mknod,rename:MEMFS.node_ops.rename,unlink:MEMFS.node_ops.unlink,rmdir:MEMFS.node_ops.rmdir,readdir:MEMFS.node_ops.readdir,symlink:MEMFS.node_ops.symlink},stream:{llseek:MEMFS.stream_ops.llseek}},file:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:{llseek:MEMFS.stream_ops.llseek,read:MEMFS.stream_ops.read,write:MEMFS.stream_ops.write,mmap:MEMFS.stream_ops.mmap,msync:MEMFS.stream_ops.msync}},link:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,readlink:MEMFS.node_ops.readlink},stream:{}},chrdev:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:FS.chrdev_stream_ops}};var node=FS.createNode(parent,name,mode,dev);if(FS.isDir(node.mode)){node.node_ops=MEMFS.ops_table.dir.node;node.stream_ops=MEMFS.ops_table.dir.stream;node.contents={}}else if(FS.isFile(node.mode)){node.node_ops=MEMFS.ops_table.file.node;node.stream_ops=MEMFS.ops_table.file.stream;node.usedBytes=0;node.contents=null}else if(FS.isLink(node.mode)){node.node_ops=MEMFS.ops_table.link.node;node.stream_ops=MEMFS.ops_table.link.stream}else if(FS.isChrdev(node.mode)){node.node_ops=MEMFS.ops_table.chrdev.node;node.stream_ops=MEMFS.ops_table.chrdev.stream}node.atime=node.mtime=node.ctime=Date.now();if(parent){parent.contents[name]=node;parent.atime=parent.mtime=parent.ctime=node.atime}return node},getFileDataAsTypedArray(node){if(!node.contents)return new Uint8Array(0);if(node.contents.subarray)return node.contents.subarray(0,node.usedBytes);return new Uint8Array(node.contents)},expandFileStorage(node,newCapacity){var prevCapacity=node.contents?node.contents.length:0;if(prevCapacity>=newCapacity)return;var CAPACITY_DOUBLING_MAX=1024*1024;newCapacity=Math.max(newCapacity,prevCapacity*(prevCapacity<CAPACITY_DOUBLING_MAX?2:1.125)>>>0);if(prevCapacity!=0)newCapacity=Math.max(newCapacity,256);var oldContents=node.contents;node.contents=new Uint8Array(newCapacity);if(node.usedBytes>0)node.contents.set(oldContents.subarray(0,node.usedBytes),0)},resizeFileStorage(node,newSize){if(node.usedBytes==newSize)return;if(newSize==0){node.contents=null;node.usedBytes=0}else{var oldContents=node.contents;node.contents=new Uint8Array(newSize);if(oldContents){node.contents.set(oldContents.subarray(0,Math.min(newSize,node.usedBytes)))}node.usedBytes=newSize}},node_ops:{getattr(node){var attr={};attr.dev=FS.isChrdev(node.mode)?node.id:1;attr.ino=node.id;attr.mode=node.mode;attr.nlink=1;attr.uid=0;attr.gid=0;attr.rdev=node.rdev;if(FS.isDir(node.mode)){attr.size=4096}else if(FS.isFile(node.mode)){attr.size=node.usedBytes}else if(FS.isLink(node.mode)){attr.size=node.link.length}else{attr.size=0}attr.atime=new Date(node.atime);attr.mtime=new Date(node.mtime);attr.ctime=new Date(node.ctime);attr.blksize=4096;attr.blocks=Math.ceil(attr.size/attr.blksize);return attr},setattr(node,attr){for(const key of["mode","atime","mtime","ctime"]){if(attr[key]!=null){node[key]=attr[key]}}if(attr.size!==undefined){MEMFS.resizeFileStorage(node,attr.size)}},lookup(parent,name){if(!MEMFS.doesNotExistError){MEMFS.doesNotExistError=new FS.ErrnoError(44);MEMFS.doesNotExistError.stack="<generic error, no stack>"}throw MEMFS.doesNotExistError},mknod(parent,name,mode,dev){return MEMFS.createNode(parent,name,mode,dev)},rename(old_node,new_dir,new_name){var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(new_node){if(FS.isDir(old_node.mode)){for(var i in new_node.contents){throw new FS.ErrnoError(55)}}FS.hashRemoveNode(new_node)}delete old_node.parent.contents[old_node.name];new_dir.contents[new_name]=old_node;old_node.name=new_name;new_dir.ctime=new_dir.mtime=old_node.parent.ctime=old_node.parent.mtime=Date.now()},unlink(parent,name){delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},rmdir(parent,name){var node=FS.lookupNode(parent,name);for(var i in node.contents){throw new FS.ErrnoError(55)}delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},readdir(node){return[".","..",...Object.keys(node.contents)]},symlink(parent,newname,oldpath){var node=MEMFS.createNode(parent,newname,511|40960,0);node.link=oldpath;return node},readlink(node){if(!FS.isLink(node.mode)){throw new FS.ErrnoError(28)}return node.link}},stream_ops:{read(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=stream.node.usedBytes)return 0;var size=Math.min(stream.node.usedBytes-position,length);if(size>8&&contents.subarray){buffer.set(contents.subarray(position,position+size),offset)}else{for(var i=0;i<size;i++)buffer[offset+i]=contents[position+i]}return size},write(stream,buffer,offset,length,position,canOwn){if(buffer.buffer===(growMemViews(),HEAP8).buffer){canOwn=false}if(!length)return 0;var node=stream.node;node.mtime=node.ctime=Date.now();if(buffer.subarray&&(!node.contents||node.contents.subarray)){if(canOwn){node.contents=buffer.subarray(offset,offset+length);node.usedBytes=length;return length}else if(node.usedBytes===0&&position===0){node.contents=buffer.slice(offset,offset+length);node.usedBytes=length;return length}else if(position+length<=node.usedBytes){node.contents.set(buffer.subarray(offset,offset+length),position);return length}}MEMFS.expandFileStorage(node,position+length);if(node.contents.subarray&&buffer.subarray){node.contents.set(buffer.subarray(offset,offset+length),position)}else{for(var i=0;i<length;i++){node.contents[position+i]=buffer[offset+i]}}node.usedBytes=Math.max(node.usedBytes,position+length);return length},llseek(stream,offset,whence){var position=offset;if(whence===1){position+=stream.position}else if(whence===2){if(FS.isFile(stream.node.mode)){position+=stream.node.usedBytes}}if(position<0){throw new FS.ErrnoError(28)}return position},mmap(stream,length,position,prot,flags){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}var ptr;var allocated;var contents=stream.node.contents;if(!(flags&2)&&contents&&contents.buffer===(growMemViews(),HEAP8).buffer){allocated=false;ptr=contents.byteOffset}else{allocated=true;ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}if(contents){if(position>0||position+length<contents.length){if(contents.subarray){contents=contents.subarray(position,position+length)}else{contents=Array.prototype.slice.call(contents,position,position+length)}}(growMemViews(),HEAP8).set(contents,ptr)}}return{ptr,allocated}},msync(stream,buffer,offset,length,mmapFlags){MEMFS.stream_ops.write(stream,buffer,0,length,offset,false);return 0}}};var FS_modeStringToFlags=str=>{var flagModes={r:0,"r+":2,w:512|64|1,"w+":512|64|2,a:1024|64|1,"a+":1024|64|2};var flags=flagModes[str];if(typeof flags=="undefined"){throw new Error(`Unknown file open mode: ${str}`)}return flags};var FS_getMode=(canRead,canWrite)=>{var mode=0;if(canRead)mode|=292|73;if(canWrite)mode|=146;return mode};var asyncLoad=async url=>{var arrayBuffer=await readAsync(url);return new Uint8Array(arrayBuffer)};var FS_createDataFile=(...args)=>FS.createDataFile(...args);var getUniqueRunDependency=id=>id;var preloadPlugins=[];var FS_handledByPreloadPlugin=async(byteArray,fullname)=>{if(typeof Browser!="undefined")Browser.init();for(var plugin of preloadPlugins){if(plugin["canHandle"](fullname)){return plugin["handle"](byteArray,fullname)}}return byteArray};var FS_preloadFile=async(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish)=>{var fullname=name?PATH_FS.resolve(PATH.join2(parent,name)):parent;var dep=getUniqueRunDependency(`cp ${fullname}`);addRunDependency(dep);try{var byteArray=url;if(typeof url=="string"){byteArray=await asyncLoad(url)}byteArray=await FS_handledByPreloadPlugin(byteArray,fullname);preFinish?.();if(!dontCreateFile){FS_createDataFile(parent,name,byteArray,canRead,canWrite,canOwn)}}finally{removeRunDependency(dep)}};var FS_createPreloadedFile=(parent,name,url,canRead,canWrite,onload,onerror,dontCreateFile,canOwn,preFinish)=>{FS_preloadFile(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish).then(onload).catch(onerror)};var FS={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,filesystems:null,syncFSRequests:0,readFiles:{},ErrnoError:class{name="ErrnoError";constructor(errno){this.errno=errno}},FSStream:class{shared={};get object(){return this.node}set object(val){this.node=val}get isRead(){return(this.flags&2097155)!==1}get isWrite(){return(this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(val){this.shared.flags=val}get position(){return this.shared.position}set position(val){this.shared.position=val}},FSNode:class{node_ops={};stream_ops={};readMode=292|73;writeMode=146;mounted=null;constructor(parent,name,mode,rdev){if(!parent){parent=this}this.parent=parent;this.mount=parent.mount;this.id=FS.nextInode++;this.name=name;this.mode=mode;this.rdev=rdev;this.atime=this.mtime=this.ctime=Date.now()}get read(){return(this.mode&this.readMode)===this.readMode}set read(val){val?this.mode|=this.readMode:this.mode&=~this.readMode}get write(){return(this.mode&this.writeMode)===this.writeMode}set write(val){val?this.mode|=this.writeMode:this.mode&=~this.writeMode}get isFolder(){return FS.isDir(this.mode)}get isDevice(){return FS.isChrdev(this.mode)}},lookupPath(path,opts={}){if(!path){throw new FS.ErrnoError(44)}opts.follow_mount??=true;if(!PATH.isAbs(path)){path=FS.cwd()+"/"+path}linkloop:for(var nlinks=0;nlinks<40;nlinks++){var parts=path.split("/").filter(p=>!!p);var current=FS.root;var current_path="/";for(var i=0;i<parts.length;i++){var islast=i===parts.length-1;if(islast&&opts.parent){break}if(parts[i]==="."){continue}if(parts[i]===".."){current_path=PATH.dirname(current_path);if(FS.isRoot(current)){path=current_path+"/"+parts.slice(i+1).join("/");nlinks--;continue linkloop}else{current=current.parent}continue}current_path=PATH.join2(current_path,parts[i]);try{current=FS.lookupNode(current,parts[i])}catch(e){if(e?.errno===44&&islast&&opts.noent_okay){return{path:current_path}}throw e}if(FS.isMountpoint(current)&&(!islast||opts.follow_mount)){current=current.mounted.root}if(FS.isLink(current.mode)&&(!islast||opts.follow)){if(!current.node_ops.readlink){throw new FS.ErrnoError(52)}var link=current.node_ops.readlink(current);if(!PATH.isAbs(link)){link=PATH.dirname(current_path)+"/"+link}path=link+"/"+parts.slice(i+1).join("/");continue linkloop}}return{path:current_path,node:current}}throw new FS.ErrnoError(32)},getPath(node){var path;while(true){if(FS.isRoot(node)){var mount=node.mount.mountpoint;if(!path)return mount;return mount[mount.length-1]!=="/"?`${mount}/${path}`:mount+path}path=path?`${node.name}/${path}`:node.name;node=node.parent}},hashName(parentid,name){var hash=0;for(var i=0;i<name.length;i++){hash=(hash<<5)-hash+name.charCodeAt(i)|0}return(parentid+hash>>>0)%FS.nameTable.length},hashAddNode(node){var hash=FS.hashName(node.parent.id,node.name);node.name_next=FS.nameTable[hash];FS.nameTable[hash]=node},hashRemoveNode(node){var hash=FS.hashName(node.parent.id,node.name);if(FS.nameTable[hash]===node){FS.nameTable[hash]=node.name_next}else{var current=FS.nameTable[hash];while(current){if(current.name_next===node){current.name_next=node.name_next;break}current=current.name_next}}},lookupNode(parent,name){var errCode=FS.mayLookup(parent);if(errCode){throw new FS.ErrnoError(errCode)}var hash=FS.hashName(parent.id,name);for(var node=FS.nameTable[hash];node;node=node.name_next){var nodeName=node.name;if(node.parent.id===parent.id&&nodeName===name){return node}}return FS.lookup(parent,name)},createNode(parent,name,mode,rdev){var node=new FS.FSNode(parent,name,mode,rdev);FS.hashAddNode(node);return node},destroyNode(node){FS.hashRemoveNode(node)},isRoot(node){return node===node.parent},isMountpoint(node){return!!node.mounted},isFile(mode){return(mode&61440)===32768},isDir(mode){return(mode&61440)===16384},isLink(mode){return(mode&61440)===40960},isChrdev(mode){return(mode&61440)===8192},isBlkdev(mode){return(mode&61440)===24576},isFIFO(mode){return(mode&61440)===4096},isSocket(mode){return(mode&49152)===49152},flagsToPermissionString(flag){var perms=["r","w","rw"][flag&3];if(flag&512){perms+="w"}return perms},nodePermissions(node,perms){if(FS.ignorePermissions){return 0}if(perms.includes("r")&&!(node.mode&292)){return 2}else if(perms.includes("w")&&!(node.mode&146)){return 2}else if(perms.includes("x")&&!(node.mode&73)){return 2}return 0},mayLookup(dir){if(!FS.isDir(dir.mode))return 54;var errCode=FS.nodePermissions(dir,"x");if(errCode)return errCode;if(!dir.node_ops.lookup)return 2;return 0},mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")},mayDelete(dir,name,isdir){var node;try{node=FS.lookupNode(dir,name)}catch(e){return e.errno}var errCode=FS.nodePermissions(dir,"wx");if(errCode){return errCode}if(isdir){if(!FS.isDir(node.mode)){return 54}if(FS.isRoot(node)||FS.getPath(node)===FS.cwd()){return 10}}else{if(FS.isDir(node.mode)){return 31}}return 0},mayOpen(node,flags){if(!node){return 44}if(FS.isLink(node.mode)){return 32}else if(FS.isDir(node.mode)){if(FS.flagsToPermissionString(flags)!=="r"||flags&(512|64)){return 31}}return FS.nodePermissions(node,FS.flagsToPermissionString(flags))},checkOpExists(op,err){if(!op){throw new FS.ErrnoError(err)}return op},MAX_OPEN_FDS:4096,nextfd(){for(var fd=0;fd<=FS.MAX_OPEN_FDS;fd++){if(!FS.streams[fd]){return fd}}throw new FS.ErrnoError(33)},getStreamChecked(fd){var stream=FS.getStream(fd);if(!stream){throw new FS.ErrnoError(8)}return stream},getStream:fd=>FS.streams[fd],createStream(stream,fd=-1){stream=Object.assign(new FS.FSStream,stream);if(fd==-1){fd=FS.nextfd()}stream.fd=fd;FS.streams[fd]=stream;return stream},closeStream(fd){FS.streams[fd]=null},dupStream(origStream,fd=-1){var stream=FS.createStream(origStream,fd);stream.stream_ops?.dup?.(stream);return stream},doSetAttr(stream,node,attr){var setattr=stream?.stream_ops.setattr;var arg=setattr?stream:node;setattr??=node.node_ops.setattr;FS.checkOpExists(setattr,63);setattr(arg,attr)},chrdev_stream_ops:{open(stream){var device=FS.getDevice(stream.node.rdev);stream.stream_ops=device.stream_ops;stream.stream_ops.open?.(stream)},llseek(){throw new FS.ErrnoError(70)}},major:dev=>dev>>8,minor:dev=>dev&255,makedev:(ma,mi)=>ma<<8|mi,registerDevice(dev,ops){FS.devices[dev]={stream_ops:ops}},getDevice:dev=>FS.devices[dev],getMounts(mount){var mounts=[];var check=[mount];while(check.length){var m=check.pop();mounts.push(m);check.push(...m.mounts)}return mounts},syncfs(populate,callback){if(typeof populate=="function"){callback=populate;populate=false}FS.syncFSRequests++;if(FS.syncFSRequests>1){err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`)}var mounts=FS.getMounts(FS.root.mount);var completed=0;function doCallback(errCode){FS.syncFSRequests--;return callback(errCode)}function done(errCode){if(errCode){if(!done.errored){done.errored=true;return doCallback(errCode)}return}if(++completed>=mounts.length){doCallback(null)}}for(var mount of mounts){if(mount.type.syncfs){mount.type.syncfs(mount,populate,done)}else{done(null)}}},mount(type,opts,mountpoint){var root=mountpoint==="/";var pseudo=!mountpoint;var node;if(root&&FS.root){throw new FS.ErrnoError(10)}else if(!root&&!pseudo){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});mountpoint=lookup.path;node=lookup.node;if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}if(!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}}var mount={type,opts,mountpoint,mounts:[]};var mountRoot=type.mount(mount);mountRoot.mount=mount;mount.root=mountRoot;if(root){FS.root=mountRoot}else if(node){node.mounted=mount;if(node.mount){node.mount.mounts.push(mount)}}return mountRoot},unmount(mountpoint){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});if(!FS.isMountpoint(lookup.node)){throw new FS.ErrnoError(28)}var node=lookup.node;var mount=node.mounted;var mounts=FS.getMounts(mount);for(var[hash,current]of Object.entries(FS.nameTable)){while(current){var next=current.name_next;if(mounts.includes(current.mount)){FS.destroyNode(current)}current=next}}node.mounted=null;var idx=node.mount.mounts.indexOf(mount);node.mount.mounts.splice(idx,1)},lookup(parent,name){return parent.node_ops.lookup(parent,name)},mknod(path,mode,dev){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);if(!name){throw new FS.ErrnoError(28)}if(name==="."||name===".."){throw new FS.ErrnoError(20)}var errCode=FS.mayCreate(parent,name);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.mknod){throw new FS.ErrnoError(63)}return parent.node_ops.mknod(parent,name,mode,dev)},statfs(path){return FS.statfsNode(FS.lookupPath(path,{follow:true}).node)},statfsStream(stream){return FS.statfsNode(stream.node)},statfsNode(node){var rtn={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:FS.nextInode,ffree:FS.nextInode-1,fsid:42,flags:2,namelen:255};if(node.node_ops.statfs){Object.assign(rtn,node.node_ops.statfs(node.mount.opts.root))}return rtn},create(path,mode=438){mode&=4095;mode|=32768;return FS.mknod(path,mode,0)},mkdir(path,mode=511){mode&=511|512;mode|=16384;return FS.mknod(path,mode,0)},mkdirTree(path,mode){var dirs=path.split("/");var d="";for(var dir of dirs){if(!dir)continue;if(d||PATH.isAbs(path))d+="/";d+=dir;try{FS.mkdir(d,mode)}catch(e){if(e.errno!=20)throw e}}},mkdev(path,mode,dev){if(typeof dev=="undefined"){dev=mode;mode=438}mode|=8192;return FS.mknod(path,mode,dev)},symlink(oldpath,newpath){if(!PATH_FS.resolve(oldpath)){throw new FS.ErrnoError(44)}var lookup=FS.lookupPath(newpath,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var newname=PATH.basename(newpath);var errCode=FS.mayCreate(parent,newname);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.symlink){throw new FS.ErrnoError(63)}return parent.node_ops.symlink(parent,newname,oldpath)},rename(old_path,new_path){var old_dirname=PATH.dirname(old_path);var new_dirname=PATH.dirname(new_path);var old_name=PATH.basename(old_path);var new_name=PATH.basename(new_path);var lookup,old_dir,new_dir;lookup=FS.lookupPath(old_path,{parent:true});old_dir=lookup.node;lookup=FS.lookupPath(new_path,{parent:true});new_dir=lookup.node;if(!old_dir||!new_dir)throw new FS.ErrnoError(44);if(old_dir.mount!==new_dir.mount){throw new FS.ErrnoError(75)}var old_node=FS.lookupNode(old_dir,old_name);var relative=PATH_FS.relative(old_path,new_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(28)}relative=PATH_FS.relative(new_path,old_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(55)}var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(old_node===new_node){return}var isdir=FS.isDir(old_node.mode);var errCode=FS.mayDelete(old_dir,old_name,isdir);if(errCode){throw new FS.ErrnoError(errCode)}errCode=new_node?FS.mayDelete(new_dir,new_name,isdir):FS.mayCreate(new_dir,new_name);if(errCode){throw new FS.ErrnoError(errCode)}if(!old_dir.node_ops.rename){throw new FS.ErrnoError(63)}if(FS.isMountpoint(old_node)||new_node&&FS.isMountpoint(new_node)){throw new FS.ErrnoError(10)}if(new_dir!==old_dir){errCode=FS.nodePermissions(old_dir,"w");if(errCode){throw new FS.ErrnoError(errCode)}}FS.hashRemoveNode(old_node);try{old_dir.node_ops.rename(old_node,new_dir,new_name);old_node.parent=new_dir}catch(e){throw e}finally{FS.hashAddNode(old_node)}},rmdir(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,true);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.rmdir){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.rmdir(parent,name);FS.destroyNode(node)},readdir(path){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var readdir=FS.checkOpExists(node.node_ops.readdir,54);return readdir(node)},unlink(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,false);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.unlink){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.unlink(parent,name);FS.destroyNode(node)},readlink(path){var lookup=FS.lookupPath(path);var link=lookup.node;if(!link){throw new FS.ErrnoError(44)}if(!link.node_ops.readlink){throw new FS.ErrnoError(28)}return link.node_ops.readlink(link)},stat(path,dontFollow){var lookup=FS.lookupPath(path,{follow:!dontFollow});var node=lookup.node;var getattr=FS.checkOpExists(node.node_ops.getattr,63);return getattr(node)},fstat(fd){var stream=FS.getStreamChecked(fd);var node=stream.node;var getattr=stream.stream_ops.getattr;var arg=getattr?stream:node;getattr??=node.node_ops.getattr;FS.checkOpExists(getattr,63);return getattr(arg)},lstat(path){return FS.stat(path,true)},doChmod(stream,node,mode,dontFollow){FS.doSetAttr(stream,node,{mode:mode&4095|node.mode&~4095,ctime:Date.now(),dontFollow})},chmod(path,mode,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChmod(null,node,mode,dontFollow)},lchmod(path,mode){FS.chmod(path,mode,true)},fchmod(fd,mode){var stream=FS.getStreamChecked(fd);FS.doChmod(stream,stream.node,mode,false)},doChown(stream,node,dontFollow){FS.doSetAttr(stream,node,{timestamp:Date.now(),dontFollow})},chown(path,uid,gid,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChown(null,node,dontFollow)},lchown(path,uid,gid){FS.chown(path,uid,gid,true)},fchown(fd,uid,gid){var stream=FS.getStreamChecked(fd);FS.doChown(stream,stream.node,false)},doTruncate(stream,node,len){if(FS.isDir(node.mode)){throw new FS.ErrnoError(31)}if(!FS.isFile(node.mode)){throw new FS.ErrnoError(28)}var errCode=FS.nodePermissions(node,"w");if(errCode){throw new FS.ErrnoError(errCode)}FS.doSetAttr(stream,node,{size:len,timestamp:Date.now()})},truncate(path,len){if(len<0){throw new FS.ErrnoError(28)}var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:true});node=lookup.node}else{node=path}FS.doTruncate(null,node,len)},ftruncate(fd,len){var stream=FS.getStreamChecked(fd);if(len<0||(stream.flags&2097155)===0){throw new FS.ErrnoError(28)}FS.doTruncate(stream,stream.node,len)},utime(path,atime,mtime){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var setattr=FS.checkOpExists(node.node_ops.setattr,63);setattr(node,{atime,mtime})},open(path,flags,mode=438){if(path===""){throw new FS.ErrnoError(44)}flags=typeof flags=="string"?FS_modeStringToFlags(flags):flags;if(flags&64){mode=mode&4095|32768}else{mode=0}var node;var isDirPath;if(typeof path=="object"){node=path}else{isDirPath=path.endsWith("/");var lookup=FS.lookupPath(path,{follow:!(flags&131072),noent_okay:true});node=lookup.node;path=lookup.path}var created=false;if(flags&64){if(node){if(flags&128){throw new FS.ErrnoError(20)}}else if(isDirPath){throw new FS.ErrnoError(31)}else{node=FS.mknod(path,mode|511,0);created=true}}if(!node){throw new FS.ErrnoError(44)}if(FS.isChrdev(node.mode)){flags&=~512}if(flags&65536&&!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}if(!created){var errCode=FS.mayOpen(node,flags);if(errCode){throw new FS.ErrnoError(errCode)}}if(flags&512&&!created){FS.truncate(node,0)}flags&=~(128|512|131072);var stream=FS.createStream({node,path:FS.getPath(node),flags,seekable:true,position:0,stream_ops:node.stream_ops,ungotten:[],error:false});if(stream.stream_ops.open){stream.stream_ops.open(stream)}if(created){FS.chmod(node,mode&511)}if(Module["logReadFiles"]&&!(flags&1)){if(!(path in FS.readFiles)){FS.readFiles[path]=1}}return stream},close(stream){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(stream.getdents)stream.getdents=null;try{if(stream.stream_ops.close){stream.stream_ops.close(stream)}}catch(e){throw e}finally{FS.closeStream(stream.fd)}stream.fd=null},isClosed(stream){return stream.fd===null},llseek(stream,offset,whence){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(!stream.seekable||!stream.stream_ops.llseek){throw new FS.ErrnoError(70)}if(whence!=0&&whence!=1&&whence!=2){throw new FS.ErrnoError(28)}stream.position=stream.stream_ops.llseek(stream,offset,whence);stream.ungotten=[];return stream.position},read(stream,buffer,offset,length,position){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.read){throw new FS.ErrnoError(28)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesRead=stream.stream_ops.read(stream,buffer,offset,length,position);if(!seeking)stream.position+=bytesRead;return bytesRead},write(stream,buffer,offset,length,position,canOwn){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===0){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.write){throw new FS.ErrnoError(28)}if(stream.seekable&&stream.flags&1024){FS.llseek(stream,0,2)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesWritten=stream.stream_ops.write(stream,buffer,offset,length,position,canOwn);if(!seeking)stream.position+=bytesWritten;return bytesWritten},mmap(stream,length,position,prot,flags){if((prot&2)!==0&&(flags&2)===0&&(stream.flags&2097155)!==2){throw new FS.ErrnoError(2)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(2)}if(!stream.stream_ops.mmap){throw new FS.ErrnoError(43)}if(!length){throw new FS.ErrnoError(28)}return stream.stream_ops.mmap(stream,length,position,prot,flags)},msync(stream,buffer,offset,length,mmapFlags){if(!stream.stream_ops.msync){return 0}return stream.stream_ops.msync(stream,buffer,offset,length,mmapFlags)},ioctl(stream,cmd,arg){if(!stream.stream_ops.ioctl){throw new FS.ErrnoError(59)}return stream.stream_ops.ioctl(stream,cmd,arg)},readFile(path,opts={}){opts.flags=opts.flags||0;opts.encoding=opts.encoding||"binary";if(opts.encoding!=="utf8"&&opts.encoding!=="binary"){abort(`Invalid encoding type "${opts.encoding}"`)}var stream=FS.open(path,opts.flags);var stat=FS.stat(path);var length=stat.size;var buf=new Uint8Array(length);FS.read(stream,buf,0,length,0);if(opts.encoding==="utf8"){buf=UTF8ArrayToString(buf)}FS.close(stream);return buf},writeFile(path,data,opts={}){opts.flags=opts.flags||577;var stream=FS.open(path,opts.flags,opts.mode);if(typeof data=="string"){data=new Uint8Array(intArrayFromString(data,true))}if(ArrayBuffer.isView(data)){FS.write(stream,data,0,data.byteLength,undefined,opts.canOwn)}else{abort("Unsupported data type")}FS.close(stream)},cwd:()=>FS.currentPath,chdir(path){var lookup=FS.lookupPath(path,{follow:true});if(lookup.node===null){throw new FS.ErrnoError(44)}if(!FS.isDir(lookup.node.mode)){throw new FS.ErrnoError(54)}var errCode=FS.nodePermissions(lookup.node,"x");if(errCode){throw new FS.ErrnoError(errCode)}FS.currentPath=lookup.path},createDefaultDirectories(){FS.mkdir("/tmp");FS.mkdir("/home");FS.mkdir("/home/web_user")},createDefaultDevices(){FS.mkdir("/dev");FS.registerDevice(FS.makedev(1,3),{read:()=>0,write:(stream,buffer,offset,length,pos)=>length,llseek:()=>0});FS.mkdev("/dev/null",FS.makedev(1,3));TTY.register(FS.makedev(5,0),TTY.default_tty_ops);TTY.register(FS.makedev(6,0),TTY.default_tty1_ops);FS.mkdev("/dev/tty",FS.makedev(5,0));FS.mkdev("/dev/tty1",FS.makedev(6,0));var randomBuffer=new Uint8Array(1024),randomLeft=0;var randomByte=()=>{if(randomLeft===0){randomFill(randomBuffer);randomLeft=randomBuffer.byteLength}return randomBuffer[--randomLeft]};FS.createDevice("/dev","random",randomByte);FS.createDevice("/dev","urandom",randomByte);FS.mkdir("/dev/shm");FS.mkdir("/dev/shm/tmp")},createSpecialDirectories(){FS.mkdir("/proc");var proc_self=FS.mkdir("/proc/self");FS.mkdir("/proc/self/fd");FS.mount({mount(){var node=FS.createNode(proc_self,"fd",16895,73);node.stream_ops={llseek:MEMFS.stream_ops.llseek};node.node_ops={lookup(parent,name){var fd=+name;var stream=FS.getStreamChecked(fd);var ret={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>stream.path},id:fd+1};ret.parent=ret;return ret},readdir(){return Array.from(FS.streams.entries()).filter(([k,v])=>v).map(([k,v])=>k.toString())}};return node}},{},"/proc/self/fd")},createStandardStreams(input,output,error){if(input){FS.createDevice("/dev","stdin",input)}else{FS.symlink("/dev/tty","/dev/stdin")}if(output){FS.createDevice("/dev","stdout",null,output)}else{FS.symlink("/dev/tty","/dev/stdout")}if(error){FS.createDevice("/dev","stderr",null,error)}else{FS.symlink("/dev/tty1","/dev/stderr")}var stdin=FS.open("/dev/stdin",0);var stdout=FS.open("/dev/stdout",1);var stderr=FS.open("/dev/stderr",1)},staticInit(){FS.nameTable=new Array(4096);FS.mount(MEMFS,{},"/");FS.createDefaultDirectories();FS.createDefaultDevices();FS.createSpecialDirectories();FS.filesystems={MEMFS}},init(input,output,error){FS.initialized=true;input??=Module["stdin"];output??=Module["stdout"];error??=Module["stderr"];FS.createStandardStreams(input,output,error)},quit(){FS.initialized=false;for(var stream of FS.streams){if(stream){FS.close(stream)}}},findObject(path,dontResolveLastLink){var ret=FS.analyzePath(path,dontResolveLastLink);if(!ret.exists){return null}return ret.object},analyzePath(path,dontResolveLastLink){try{var lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});path=lookup.path}catch(e){}var ret={isRoot:false,exists:false,error:0,name:null,path:null,object:null,parentExists:false,parentPath:null,parentObject:null};try{var lookup=FS.lookupPath(path,{parent:true});ret.parentExists=true;ret.parentPath=lookup.path;ret.parentObject=lookup.node;ret.name=PATH.basename(path);lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});ret.exists=true;ret.path=lookup.path;ret.object=lookup.node;ret.name=lookup.node.name;ret.isRoot=lookup.path==="/"}catch(e){ret.error=e.errno}return ret},createPath(parent,path,canRead,canWrite){parent=typeof parent=="string"?parent:FS.getPath(parent);var parts=path.split("/").reverse();while(parts.length){var part=parts.pop();if(!part)continue;var current=PATH.join2(parent,part);try{FS.mkdir(current)}catch(e){if(e.errno!=20)throw e}parent=current}return current},createFile(parent,name,properties,canRead,canWrite){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(canRead,canWrite);return FS.create(path,mode)},createDataFile(parent,name,data,canRead,canWrite,canOwn){var path=name;if(parent){parent=typeof parent=="string"?parent:FS.getPath(parent);path=name?PATH.join2(parent,name):parent}var mode=FS_getMode(canRead,canWrite);var node=FS.create(path,mode);if(data){if(typeof data=="string"){var arr=new Array(data.length);for(var i=0,len=data.length;i<len;++i)arr[i]=data.charCodeAt(i);data=arr}FS.chmod(node,mode|146);var stream=FS.open(node,577);FS.write(stream,data,0,data.length,0,canOwn);FS.close(stream);FS.chmod(node,mode)}},createDevice(parent,name,input,output){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(!!input,!!output);FS.createDevice.major??=64;var dev=FS.makedev(FS.createDevice.major++,0);FS.registerDevice(dev,{open(stream){stream.seekable=false},close(stream){if(output?.buffer?.length){output(10)}},read(stream,buffer,offset,length,pos){var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=input()}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){for(var i=0;i<length;i++){try{output(buffer[offset+i])}catch(e){throw new FS.ErrnoError(29)}}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}});return FS.mkdev(path,mode,dev)},forceLoadFile(obj){if(obj.isDevice||obj.isFolder||obj.link||obj.contents)return true;if(globalThis.XMLHttpRequest){abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.")}else{try{obj.contents=readBinary(obj.url)}catch(e){throw new FS.ErrnoError(29)}}},createLazyFile(parent,name,url,canRead,canWrite){class LazyUint8Array{lengthKnown=false;chunks=[];get(idx){if(idx>this.length-1||idx<0){return undefined}var chunkOffset=idx%this.chunkSize;var chunkNum=idx/this.chunkSize|0;return this.getter(chunkNum)[chunkOffset]}setDataGetter(getter){this.getter=getter}cacheLength(){var xhr=new XMLHttpRequest;xhr.open("HEAD",url,false);xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);var datalength=Number(xhr.getResponseHeader("Content-length"));var header;var hasByteServing=(header=xhr.getResponseHeader("Accept-Ranges"))&&header==="bytes";var usesGzip=(header=xhr.getResponseHeader("Content-Encoding"))&&header==="gzip";var chunkSize=1024*1024;if(!hasByteServing)chunkSize=datalength;var doXHR=(from,to)=>{if(from>to)abort("invalid range ("+from+", "+to+") or no bytes requested!");if(to>datalength-1)abort("only "+datalength+" bytes available! programmer error!");var xhr=new XMLHttpRequest;xhr.open("GET",url,false);if(datalength!==chunkSize)xhr.setRequestHeader("Range","bytes="+from+"-"+to);xhr.responseType="arraybuffer";if(xhr.overrideMimeType){xhr.overrideMimeType("text/plain; charset=x-user-defined")}xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);if(xhr.response!==undefined){return new Uint8Array(xhr.response||[])}return intArrayFromString(xhr.responseText||"",true)};var lazyArray=this;lazyArray.setDataGetter(chunkNum=>{var start=chunkNum*chunkSize;var end=(chunkNum+1)*chunkSize-1;end=Math.min(end,datalength-1);if(typeof lazyArray.chunks[chunkNum]=="undefined"){lazyArray.chunks[chunkNum]=doXHR(start,end)}if(typeof lazyArray.chunks[chunkNum]=="undefined")abort("doXHR failed!");return lazyArray.chunks[chunkNum]});if(usesGzip||!datalength){chunkSize=datalength=1;datalength=this.getter(0).length;chunkSize=datalength;out("LazyFiles on gzip forces download of the whole file when length is accessed")}this._length=datalength;this._chunkSize=chunkSize;this.lengthKnown=true}get length(){if(!this.lengthKnown){this.cacheLength()}return this._length}get chunkSize(){if(!this.lengthKnown){this.cacheLength()}return this._chunkSize}}if(globalThis.XMLHttpRequest){if(!ENVIRONMENT_IS_WORKER)abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");var lazyArray=new LazyUint8Array;var properties={isDevice:false,contents:lazyArray}}else{var properties={isDevice:false,url}}var node=FS.createFile(parent,name,properties,canRead,canWrite);if(properties.contents){node.contents=properties.contents}else if(properties.url){node.contents=null;node.url=properties.url}Object.defineProperties(node,{usedBytes:{get:function(){return this.contents.length}}});var stream_ops={};for(const[key,fn]of Object.entries(node.stream_ops)){stream_ops[key]=(...args)=>{FS.forceLoadFile(node);return fn(...args)}}function writeChunks(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=contents.length)return 0;var size=Math.min(contents.length-position,length);if(contents.slice){for(var i=0;i<size;i++){buffer[offset+i]=contents[position+i]}}else{for(var i=0;i<size;i++){buffer[offset+i]=contents.get(position+i)}}return size}stream_ops.read=(stream,buffer,offset,length,position)=>{FS.forceLoadFile(node);return writeChunks(stream,buffer,offset,length,position)};stream_ops.mmap=(stream,length,position,prot,flags)=>{FS.forceLoadFile(node);var ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}writeChunks(stream,(growMemViews(),HEAP8),ptr,length,position);return{ptr,allocated:true}};node.stream_ops=stream_ops;return node}};var UTF8ToString=(ptr,maxBytesToRead,ignoreNul)=>ptr?UTF8ArrayToString((growMemViews(),HEAPU8),ptr,maxBytesToRead,ignoreNul):"";var SYSCALLS={DEFAULT_POLLMASK:5,calculateAt(dirfd,path,allowEmpty){if(PATH.isAbs(path)){return path}var dir;if(dirfd===-100){dir=FS.cwd()}else{var dirstream=SYSCALLS.getStreamFromFD(dirfd);dir=dirstream.path}if(path.length==0){if(!allowEmpty){throw new FS.ErrnoError(44)}return dir}return dir+"/"+path},writeStat(buf,stat){(growMemViews(),HEAPU32)[buf/4]=stat.dev;(growMemViews(),HEAPU32)[(buf+4)/4]=stat.mode;(growMemViews(),HEAPU64)[(buf+8)/8]=BigInt(stat.nlink);(growMemViews(),HEAPU32)[(buf+16)/4]=stat.uid;(growMemViews(),HEAPU32)[(buf+20)/4]=stat.gid;(growMemViews(),HEAPU32)[(buf+24)/4]=stat.rdev;(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stat.size);(growMemViews(),HEAP32)[(buf+40)/4]=4096;(growMemViews(),HEAP32)[(buf+44)/4]=stat.blocks;var atime=stat.atime.getTime();var mtime=stat.mtime.getTime();var ctime=stat.ctime.getTime();(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(Math.floor(atime/1e3));(growMemViews(),HEAPU64)[(buf+56)/8]=BigInt(atime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+64)/8]=BigInt(Math.floor(mtime/1e3));(growMemViews(),HEAPU64)[(buf+72)/8]=BigInt(mtime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+80)/8]=BigInt(Math.floor(ctime/1e3));(growMemViews(),HEAPU64)[(buf+88)/8]=BigInt(ctime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+96)/8]=BigInt(stat.ino);return 0},writeStatFs(buf,stats){(growMemViews(),HEAPU32)[(buf+8)/4]=stats.bsize;(growMemViews(),HEAPU32)[(buf+72)/4]=stats.bsize;(growMemViews(),HEAP64)[(buf+16)/8]=BigInt(stats.blocks);(growMemViews(),HEAP64)[(buf+24)/8]=BigInt(stats.bfree);(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stats.bavail);(growMemViews(),HEAP64)[(buf+40)/8]=BigInt(stats.files);(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(stats.ffree);(growMemViews(),HEAPU32)[(buf+56)/4]=stats.fsid;(growMemViews(),HEAPU32)[(buf+80)/4]=stats.flags;(growMemViews(),HEAPU32)[(buf+64)/4]=stats.namelen},doMsync(addr,stream,len,flags,offset){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}if(flags&2){return 0}var buffer=(growMemViews(),HEAPU8).slice(addr,addr+len);FS.msync(stream,buffer,offset,len,flags)},getStreamFromFD(fd){var stream=FS.getStreamChecked(fd);return stream},varargs:undefined,getStr(ptr){var ret=UTF8ToString(ptr);return ret}};function ___syscall_fcntl64(fd,cmd,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(3,0,1,fd,cmd,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(cmd){case 0:{var arg=syscallGetVarargI();if(arg<0){return-28}while(FS.streams[arg]){arg++}var newStream;newStream=FS.dupStream(stream,arg);return newStream.fd}case 1:case 2:return 0;case 3:return stream.flags;case 4:{var arg=syscallGetVarargI();stream.flags|=arg;return 0}case 5:{var arg=syscallGetVarargP();var offset=0;(growMemViews(),HEAP16)[(arg+offset)/2]=2;return 0}case 6:case 7:return 0}return-28}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_fstat64(fd,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(4,0,1,fd,buf);buf=bigintToI53Checked(buf);try{return SYSCALLS.writeStat(buf,FS.fstat(fd))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var stringToUTF8=(str,outPtr,maxBytesToWrite)=>stringToUTF8Array(str,(growMemViews(),HEAPU8),outPtr,maxBytesToWrite);function ___syscall_getcwd(buf,size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(5,0,1,buf,size);buf=bigintToI53Checked(buf);size=bigintToI53Checked(size);try{if(size===0)return-28;var cwd=FS.cwd();var cwdLengthInBytes=lengthBytesUTF8(cwd)+1;if(size<cwdLengthInBytes)return-68;stringToUTF8(cwd,buf,size);return cwdLengthInBytes}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_getdents64(fd,dirp,count){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(6,0,1,fd,dirp,count);dirp=bigintToI53Checked(dirp);count=bigintToI53Checked(count);try{var stream=SYSCALLS.getStreamFromFD(fd);stream.getdents||=FS.readdir(stream.path);var struct_size=280;var pos=0;var off=FS.llseek(stream,0,1);var startIdx=Math.floor(off/struct_size);var endIdx=Math.min(stream.getdents.length,startIdx+Math.floor(count/struct_size));for(var idx=startIdx;idx<endIdx;idx++){var id;var type;var name=stream.getdents[idx];if(name==="."){id=stream.node.id;type=4}else if(name===".."){var lookup=FS.lookupPath(stream.path,{parent:true});id=lookup.node.id;type=4}else{var child;try{child=FS.lookupNode(stream.node,name)}catch(e){if(e?.errno===28){continue}throw e}id=child.id;type=FS.isChrdev(child.mode)?2:FS.isDir(child.mode)?4:FS.isLink(child.mode)?10:8}(growMemViews(),HEAP64)[(dirp+pos)/8]=BigInt(id);(growMemViews(),HEAP64)[(dirp+pos+8)/8]=BigInt((idx+1)*struct_size);(growMemViews(),HEAP16)[(dirp+pos+16)/2]=280;(growMemViews(),HEAP8)[dirp+pos+18]=type;stringToUTF8(name,dirp+pos+19,256);pos+=struct_size}FS.llseek(stream,idx*struct_size,0);return pos}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_ioctl(fd,op,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(7,0,1,fd,op,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(op){case 21509:{if(!stream.tty)return-59;return 0}case 21505:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcgets){var termios=stream.tty.ops.ioctl_tcgets(stream);var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=termios.c_iflag||0;(growMemViews(),HEAP32)[(argp+4)/4]=termios.c_oflag||0;(growMemViews(),HEAP32)[(argp+8)/4]=termios.c_cflag||0;(growMemViews(),HEAP32)[(argp+12)/4]=termios.c_lflag||0;for(var i=0;i<32;i++){(growMemViews(),HEAP8)[argp+i+17]=termios.c_cc[i]||0}return 0}return 0}case 21510:case 21511:case 21512:{if(!stream.tty)return-59;return 0}case 21506:case 21507:case 21508:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcsets){var argp=syscallGetVarargP();var c_iflag=(growMemViews(),HEAP32)[argp/4];var c_oflag=(growMemViews(),HEAP32)[(argp+4)/4];var c_cflag=(growMemViews(),HEAP32)[(argp+8)/4];var c_lflag=(growMemViews(),HEAP32)[(argp+12)/4];var c_cc=[];for(var i=0;i<32;i++){c_cc.push((growMemViews(),HEAP8)[argp+i+17])}return stream.tty.ops.ioctl_tcsets(stream.tty,op,{c_iflag,c_oflag,c_cflag,c_lflag,c_cc})}return 0}case 21519:{if(!stream.tty)return-59;var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=0;return 0}case 21520:{if(!stream.tty)return-59;return-28}case 21537:case 21531:{var argp=syscallGetVarargP();return FS.ioctl(stream,op,argp)}case 21523:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tiocgwinsz){var winsize=stream.tty.ops.ioctl_tiocgwinsz(stream.tty);var argp=syscallGetVarargP();(growMemViews(),HEAP16)[argp/2]=winsize[0];(growMemViews(),HEAP16)[(argp+2)/2]=winsize[1]}return 0}case 21524:{if(!stream.tty)return-59;return 0}case 21515:{if(!stream.tty)return-59;return 0}default:return-28}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_lstat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(8,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.lstat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_newfstatat(dirfd,path,buf,flags){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(9,0,1,dirfd,path,buf,flags);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);var nofollow=flags&256;var allowEmpty=flags&4096;flags=flags&~6400;path=SYSCALLS.calculateAt(dirfd,path,allowEmpty);return SYSCALLS.writeStat(buf,nofollow?FS.lstat(path):FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_openat(dirfd,path,flags,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(10,0,1,dirfd,path,flags,varargs);path=bigintToI53Checked(path);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{path=SYSCALLS.getStr(path);path=SYSCALLS.calculateAt(dirfd,path);var mode=varargs?syscallGetVarargI():0;return FS.open(path,flags,mode).fd}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_stat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(11,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __abort_js=()=>abort("");function __emscripten_init_main_thread_js(tb){tb=bigintToI53Checked(tb);__emscripten_thread_init(tb,!ENVIRONMENT_IS_WORKER,1,!ENVIRONMENT_IS_WEB,5242880,false);PThread.threadInitTLS()}var handleException=e=>{if(e instanceof ExitStatus||e=="unwind"){return EXITSTATUS}quit_(1,e)};var maybeExit=()=>{if(!keepRuntimeAlive()){try{if(ENVIRONMENT_IS_PTHREAD){if(_pthread_self())__emscripten_thread_exit(EXITSTATUS);return}_exit(EXITSTATUS)}catch(e){handleException(e)}}};var callUserCallback=func=>{if(ABORT){return}try{func();maybeExit()}catch(e){handleException(e)}};function __emscripten_thread_mailbox_await(pthread_ptr){pthread_ptr=bigintToI53Checked(pthread_ptr);if(Atomics.waitAsync){var wait=Atomics.waitAsync((growMemViews(),HEAP32),pthread_ptr/4,pthread_ptr);wait.value.then(checkMailbox);var waitingAsync=pthread_ptr+228;Atomics.store((growMemViews(),HEAP32),waitingAsync/4,1)}}var checkMailbox=()=>callUserCallback(()=>{var pthread_ptr=_pthread_self();if(pthread_ptr){__emscripten_thread_mailbox_await(pthread_ptr);__emscripten_check_mailbox()}});function __emscripten_notify_mailbox_postmessage(targetThread,currThreadId){targetThread=bigintToI53Checked(targetThread);currThreadId=bigintToI53Checked(currThreadId);if(targetThread==currThreadId){setTimeout(checkMailbox)}else if(ENVIRONMENT_IS_PTHREAD){postMessage({targetThread,cmd:"checkMailbox"})}else{var worker=PThread.pthreads[targetThread];if(!worker){return}worker.postMessage({cmd:"checkMailbox"})}}var proxiedJSCallArgs=[];function __emscripten_receive_on_main_thread_js(funcIndex,emAsmAddr,callingThread,numCallArgs,args){emAsmAddr=bigintToI53Checked(emAsmAddr);callingThread=bigintToI53Checked(callingThread);args=bigintToI53Checked(args);numCallArgs/=2;proxiedJSCallArgs.length=numCallArgs;var b=args/8;for(var i=0;i<numCallArgs;i++){if((growMemViews(),HEAP64)[b+2*i]){proxiedJSCallArgs[i]=(growMemViews(),HEAP64)[b+2*i+1]}else{proxiedJSCallArgs[i]=(growMemViews(),HEAPF64)[b+2*i+1]}}var func=proxiedFunctionTable[funcIndex];PThread.currentProxiedOperationCallerThread=callingThread;var rtn=func(...proxiedJSCallArgs);PThread.currentProxiedOperationCallerThread=0;if(typeof rtn=="bigint"){rtn=bigintToI53Checked(rtn)}return rtn}function __emscripten_thread_cleanup(thread){thread=bigintToI53Checked(thread);if(!ENVIRONMENT_IS_PTHREAD)cleanupThread(thread);else postMessage({cmd:"cleanupThread",thread})}function __emscripten_thread_set_strongref(thread){thread=bigintToI53Checked(thread);if(ENVIRONMENT_IS_NODE){PThread.pthreads[thread].ref()}}var isLeapYear=year=>year%4===0&&(year%100!==0||year%400===0);var MONTH_DAYS_LEAP_CUMULATIVE=[0,31,60,91,121,152,182,213,244,274,305,335];var MONTH_DAYS_REGULAR_CUMULATIVE=[0,31,59,90,120,151,181,212,243,273,304,334];var ydayFromDate=date=>{var leap=isLeapYear(date.getFullYear());var monthDaysCumulative=leap?MONTH_DAYS_LEAP_CUMULATIVE:MONTH_DAYS_REGULAR_CUMULATIVE;var yday=monthDaysCumulative[date.getMonth()]+date.getDate()-1;return yday};function __localtime_js(time,tmPtr){time=bigintToI53Checked(time);tmPtr=bigintToI53Checked(tmPtr);var date=new Date(time*1e3);(growMemViews(),HEAP32)[tmPtr/4]=date.getSeconds();(growMemViews(),HEAP32)[(tmPtr+4)/4]=date.getMinutes();(growMemViews(),HEAP32)[(tmPtr+8)/4]=date.getHours();(growMemViews(),HEAP32)[(tmPtr+12)/4]=date.getDate();(growMemViews(),HEAP32)[(tmPtr+16)/4]=date.getMonth();(growMemViews(),HEAP32)[(tmPtr+20)/4]=date.getFullYear()-1900;(growMemViews(),HEAP32)[(tmPtr+24)/4]=date.getDay();var yday=ydayFromDate(date)|0;(growMemViews(),HEAP32)[(tmPtr+28)/4]=yday;(growMemViews(),HEAP64)[(tmPtr+40)/8]=BigInt(-(date.getTimezoneOffset()*60));var start=new Date(date.getFullYear(),0,1);var summerOffset=new Date(date.getFullYear(),6,1).getTimezoneOffset();var winterOffset=start.getTimezoneOffset();var dst=(summerOffset!=winterOffset&&date.getTimezoneOffset()==Math.min(winterOffset,summerOffset))|0;(growMemViews(),HEAP32)[(tmPtr+32)/4]=dst}function __mmap_js(len,prot,flags,fd,offset,allocated,addr){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(12,0,1,len,prot,flags,fd,offset,allocated,addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);allocated=bigintToI53Checked(allocated);addr=bigintToI53Checked(addr);try{var stream=SYSCALLS.getStreamFromFD(fd);var res=FS.mmap(stream,len,offset,prot,flags);var ptr=res.ptr;(growMemViews(),HEAP32)[allocated/4]=res.allocated;(growMemViews(),HEAPU64)[addr/8]=BigInt(ptr);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function __munmap_js(addr,len,prot,flags,fd,offset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(13,0,1,addr,len,prot,flags,fd,offset);addr=bigintToI53Checked(addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);try{var stream=SYSCALLS.getStreamFromFD(fd);if(prot&2){SYSCALLS.doMsync(addr,stream,len,flags,offset)}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __tzset_js=function(timezone,daylight,std_name,dst_name){timezone=bigintToI53Checked(timezone);daylight=bigintToI53Checked(daylight);std_name=bigintToI53Checked(std_name);dst_name=bigintToI53Checked(dst_name);var currentYear=(new Date).getFullYear();var winter=new Date(currentYear,0,1);var summer=new Date(currentYear,6,1);var winterOffset=winter.getTimezoneOffset();var summerOffset=summer.getTimezoneOffset();var stdTimezoneOffset=Math.max(winterOffset,summerOffset);(growMemViews(),HEAPU64)[timezone/8]=BigInt(stdTimezoneOffset*60);(growMemViews(),HEAP32)[daylight/4]=Number(winterOffset!=summerOffset);var extractZone=timezoneOffset=>{var sign=timezoneOffset>=0?"-":"+";var absOffset=Math.abs(timezoneOffset);var hours=String(Math.floor(absOffset/60)).padStart(2,"0");var minutes=String(absOffset%60).padStart(2,"0");return`UTC${sign}${hours}${minutes}`};var winterName=extractZone(winterOffset);var summerName=extractZone(summerOffset);if(summerOffset<winterOffset){stringToUTF8(winterName,std_name,17);stringToUTF8(summerName,dst_name,17)}else{stringToUTF8(winterName,dst_name,17);stringToUTF8(summerName,std_name,17)}};var _emscripten_get_now=()=>performance.timeOrigin+performance.now();var _emscripten_date_now=()=>Date.now();var nowIsMonotonic=1;var checkWasiClock=clock_id=>clock_id>=0&&clock_id<=3;function _clock_time_get(clk_id,ignored_precision,ptime){ignored_precision=bigintToI53Checked(ignored_precision);ptime=bigintToI53Checked(ptime);if(!checkWasiClock(clk_id)){return 28}var now;if(clk_id===0){now=_emscripten_date_now()}else if(nowIsMonotonic){now=_emscripten_get_now()}else{return 52}var nsec=Math.round(now*1e3*1e3);(growMemViews(),HEAP64)[ptime/8]=BigInt(nsec);return 0}var _emscripten_check_blocking_allowed=()=>{};var runtimeKeepalivePush=()=>{runtimeKeepaliveCounter+=1};var _emscripten_exit_with_live_runtime=()=>{runtimeKeepalivePush();throw"unwind"};var jsStackTrace=()=>(new Error).stack.toString();var getCallstack=flags=>{var callstack=jsStackTrace();var lines=callstack.split("\\n");callstack="";var firefoxRe=new RegExp("\\\\s*(.*?)@(.*?):([0-9]+):([0-9]+)");var chromeRe=new RegExp("\\\\s*at (.*?) \\\\((.*):(.*):(.*)\\\\)");for(var line of lines){var symbolName="";var file="";var lineno=0;var column=0;var parts=chromeRe.exec(line);if(parts?.length==5){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]}else{parts=firefoxRe.exec(line);if(parts?.length>=4){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]|0}else{callstack+=line+"\\n";continue}}if(symbolName=="_emscripten_log"||symbolName=="_emscripten_get_callstack"){callstack="";continue}if(flags&24){if(flags&64){file=file.substring(file.replace(/\\\\/g,"/").lastIndexOf("/")+1)}callstack+=`    at ${symbolName} (${file}:${lineno}:${column})\\n`}}callstack=callstack.replace(/\\s+$/,"");return callstack};function _emscripten_get_callstack(flags,str,maxbytes){str=bigintToI53Checked(str);var callstack=getCallstack(flags);if(!str||maxbytes<=0){return lengthBytesUTF8(callstack)+1}var bytesWrittenExcludingNull=stringToUTF8(callstack,str,maxbytes);return bytesWrittenExcludingNull+1}var getHeapMax=()=>4294967296;var _emscripten_get_heap_max=()=>BigInt(getHeapMax());var _emscripten_has_asyncify=()=>2;var _emscripten_num_logical_cores=()=>ENVIRONMENT_IS_NODE?require("os").cpus().length:navigator["hardwareConcurrency"];var growMemory=size=>{var oldHeapSize=wasmMemory.buffer.byteLength;var pages=(size-oldHeapSize+65535)/65536|0;try{wasmMemory.grow(BigInt(pages));updateMemoryViews();return 1}catch(e){}};function _emscripten_resize_heap(requestedSize){requestedSize=bigintToI53Checked(requestedSize);var oldSize=(growMemViews(),HEAPU8).length;if(requestedSize<=oldSize){return false}var maxHeapSize=getHeapMax();if(requestedSize>maxHeapSize){return false}for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignMemory(Math.max(requestedSize,overGrownHeapSize),65536));var replacement=growMemory(newSize);if(replacement){return true}}return false}var stringToUTF8OnStack=str=>{var size=lengthBytesUTF8(str)+1;var ret=stackAlloc(size);stringToUTF8(str,ret,size);return ret};var writeI53ToI64=(ptr,num)=>{(growMemViews(),HEAPU32)[ptr/4]=num;var lower=(growMemViews(),HEAPU32)[ptr/4];(growMemViews(),HEAPU32)[(ptr+4)/4]=(num-lower)/4294967296};var stringToNewUTF8=str=>{var size=lengthBytesUTF8(str)+1;var ret=_malloc(size);if(ret)stringToUTF8(str,ret,size);return ret};var readI53FromI64=ptr=>(growMemViews(),HEAPU32)[ptr/4]+(growMemViews(),HEAP32)[(ptr+4)/4]*4294967296;var WebGPU={Internals:{jsObjects:[],jsObjectInsert:(ptr,jsObject)=>{WebGPU.Internals.jsObjects[ptr]=jsObject},bufferOnUnmaps:[],futures:[],futureInsert:(futureId,promise)=>{WebGPU.Internals.futures[futureId]=new Promise(resolve=>promise.finally(()=>resolve(futureId)))}},getJsObject:ptr=>{if(!ptr)return undefined;return WebGPU.Internals.jsObjects[ptr]},importJsAdapter:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateAdapter(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroup:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroup(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroupLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroupLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBuffer:(buffer,parentPtr=0)=>{assert(buffer.mapState==="unmapped");var bufferPtr=_emwgpuCreateBuffer(parentPtr);WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);return bufferPtr},importJsCommandBuffer:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandBuffer(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsCommandEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsDevice:(device,parentPtr=0)=>{var queuePtr=_emwgpuCreateQueue(parentPtr);var devicePtr=_emwgpuCreateDevice(parentPtr,queuePtr);WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);return devicePtr},importJsExternalTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateExternalTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsPipelineLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreatePipelineLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQuerySet:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQuerySet(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQueue:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQueue(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundle:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundle(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundleEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundleEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSampler:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSampler(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsShaderModule:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateShaderModule(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSurface:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSurface(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTextureView:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTextureView(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},errorCallback:(callback,type,message,userdata)=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(message);((a1,a2,a3)=>getWasmTableEntry(callback).call(null,a1,BigInt(a2),BigInt(a3)))(type,BigInt(messagePtr),userdata);stackRestore(sp)},iterateExtensions:(root,handlers)=>{for(var ptr=Number((growMemViews(),HEAPU64)[root/8]);ptr;ptr=Number((growMemViews(),HEAPU64)[ptr/8])){var sType=(growMemViews(),HEAP32)[(ptr+8)/4];var handler=handlers[sType](ptr)}},setStringView:(ptr,data,length)=>{(growMemViews(),HEAPU64)[ptr/8]=BigInt(data);(growMemViews(),HEAPU64)[(ptr+8)/8]=BigInt(length)},makeStringFromStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);return UTF8ToString(ptr,length)},makeStringFromOptionalStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);if(!ptr){if(length===0){return""}return undefined}return UTF8ToString(ptr,length)},makeColor:ptr=>({r:(growMemViews(),HEAPF64)[ptr/8],g:(growMemViews(),HEAPF64)[(ptr+8)/8],b:(growMemViews(),HEAPF64)[(ptr+16)/8],a:(growMemViews(),HEAPF64)[(ptr+24)/8]}),makeExtent3D:ptr=>({width:(growMemViews(),HEAPU32)[ptr/4],height:(growMemViews(),HEAPU32)[(ptr+4)/4],depthOrArrayLayers:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeOrigin3D:ptr=>({x:(growMemViews(),HEAPU32)[ptr/4],y:(growMemViews(),HEAPU32)[(ptr+4)/4],z:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeTexelCopyTextureInfo:ptr=>({texture:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[ptr/8])),mipLevel:(growMemViews(),HEAPU32)[(ptr+8)/4],origin:WebGPU.makeOrigin3D(ptr+12),aspect:WebGPU.TextureAspect[(growMemViews(),HEAP32)[(ptr+24)/4]]}),makeTexelCopyBufferLayout:ptr=>{var bytesPerRow=(growMemViews(),HEAPU32)[(ptr+8)/4];var rowsPerImage=(growMemViews(),HEAPU32)[(ptr+12)/4];return{offset:readI53FromI64(ptr),bytesPerRow:bytesPerRow===4294967295?undefined:bytesPerRow,rowsPerImage:rowsPerImage===4294967295?undefined:rowsPerImage}},makeTexelCopyBufferInfo:ptr=>{var layoutPtr=ptr+0;var bufferCopyView=WebGPU.makeTexelCopyBufferLayout(layoutPtr);bufferCopyView["buffer"]=WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+16)/8]));return bufferCopyView},makePassTimestampWrites:ptr=>{if(ptr===0)return undefined;return{querySet:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),beginningOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+16)/4],endOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+20)/4]}},makePipelineConstants:(constantCount,constantsPtr)=>{if(!constantCount)return;var constants={};for(var i=0;i<constantCount;++i){var entryPtr=constantsPtr+32*i;var key=WebGPU.makeStringFromStringView(entryPtr+8);constants[key]=(growMemViews(),HEAPF64)[(entryPtr+24)/8]}return constants},makePipelineLayout:layoutPtr=>{if(!layoutPtr)return"auto";return WebGPU.getJsObject(layoutPtr)},makeComputeState:ptr=>{if(!ptr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(ptr+32)/8]),Number((growMemViews(),HEAPU64)[(ptr+40)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(ptr+16)};return desc},makeComputePipelineDesc:descriptor=>{var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),compute:WebGPU.makeComputeState(descriptor+32)};return desc},makeRenderPipelineDesc:descriptor=>{function makePrimitiveState(psPtr){if(!psPtr)return undefined;return{topology:WebGPU.PrimitiveTopology[(growMemViews(),HEAP32)[(psPtr+8)/4]],stripIndexFormat:WebGPU.IndexFormat[(growMemViews(),HEAP32)[(psPtr+12)/4]],frontFace:WebGPU.FrontFace[(growMemViews(),HEAP32)[(psPtr+16)/4]],cullMode:WebGPU.CullMode[(growMemViews(),HEAP32)[(psPtr+20)/4]],unclippedDepth:!!(growMemViews(),HEAPU32)[(psPtr+24)/4]}}function makeBlendComponent(bdPtr){if(!bdPtr)return undefined;return{operation:WebGPU.BlendOperation[(growMemViews(),HEAP32)[bdPtr/4]],srcFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+4)/4]],dstFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+8)/4]]}}function makeBlendState(bsPtr){if(!bsPtr)return undefined;return{alpha:makeBlendComponent(bsPtr+12),color:makeBlendComponent(bsPtr+0)}}function makeColorState(csPtr){var format=WebGPU.TextureFormat[(growMemViews(),HEAP32)[(csPtr+8)/4]];return format?{format,blend:makeBlendState(Number((growMemViews(),HEAPU64)[(csPtr+16)/8])),writeMask:(growMemViews(),HEAPU32)[(csPtr+24)/4]}:undefined}function makeColorStates(count,csArrayPtr){var states=[];for(var i=0;i<count;++i){states.push(makeColorState(csArrayPtr+32*i))}return states}function makeStencilStateFace(ssfPtr){return{compare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[ssfPtr/4]],failOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+4)/4]],depthFailOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+8)/4]],passOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+12)/4]]}}function makeDepthStencilState(dssPtr){if(!dssPtr)return undefined;return{format:WebGPU.TextureFormat[(growMemViews(),HEAP32)[(dssPtr+8)/4]],depthWriteEnabled:!!(growMemViews(),HEAPU32)[(dssPtr+12)/4],depthCompare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[(dssPtr+16)/4]],stencilFront:makeStencilStateFace(dssPtr+20),stencilBack:makeStencilStateFace(dssPtr+36),stencilReadMask:(growMemViews(),HEAPU32)[(dssPtr+52)/4],stencilWriteMask:(growMemViews(),HEAPU32)[(dssPtr+56)/4],depthBias:(growMemViews(),HEAP32)[(dssPtr+60)/4],depthBiasSlopeScale:(growMemViews(),HEAPF32)[(dssPtr+64)/4],depthBiasClamp:(growMemViews(),HEAPF32)[(dssPtr+68)/4]}}function makeVertexAttribute(vaPtr){return{format:WebGPU.VertexFormat[(growMemViews(),HEAP32)[(vaPtr+8)/4]],offset:readI53FromI64(vaPtr+16),shaderLocation:(growMemViews(),HEAPU32)[(vaPtr+24)/4]}}function makeVertexAttributes(count,vaArrayPtr){var vas=[];for(var i=0;i<count;++i){vas.push(makeVertexAttribute(vaArrayPtr+i*32))}return vas}function makeVertexBuffer(vbPtr){if(!vbPtr)return undefined;var stepMode=WebGPU.VertexStepMode[(growMemViews(),HEAP32)[(vbPtr+8)/4]];var attributeCount=Number((growMemViews(),HEAPU64)[(vbPtr+24)/8]);if(!stepMode&&!attributeCount){return null}return{arrayStride:readI53FromI64(vbPtr+16),stepMode,attributes:makeVertexAttributes(attributeCount,Number((growMemViews(),HEAPU64)[(vbPtr+32)/8]))}}function makeVertexBuffers(count,vbArrayPtr){if(!count)return undefined;var vbs=[];for(var i=0;i<count;++i){vbs.push(makeVertexBuffer(vbArrayPtr+i*40))}return vbs}function makeVertexState(viPtr){if(!viPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(viPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(viPtr+32)/8]),Number((growMemViews(),HEAPU64)[(viPtr+40)/8])),buffers:makeVertexBuffers(Number((growMemViews(),HEAPU64)[(viPtr+48)/8]),Number((growMemViews(),HEAPU64)[(viPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(viPtr+16)};return desc}function makeMultisampleState(msPtr){if(!msPtr)return undefined;return{count:(growMemViews(),HEAPU32)[(msPtr+8)/4],mask:(growMemViews(),HEAPU32)[(msPtr+12)/4],alphaToCoverageEnabled:!!(growMemViews(),HEAPU32)[(msPtr+16)/4]}}function makeFragmentState(fsPtr){if(!fsPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(fsPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(fsPtr+32)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+40)/8])),targets:makeColorStates(Number((growMemViews(),HEAPU64)[(fsPtr+48)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(fsPtr+16)};return desc}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),vertex:makeVertexState(descriptor+32),primitive:makePrimitiveState(descriptor+96),depthStencil:makeDepthStencilState(Number((growMemViews(),HEAPU64)[(descriptor+128)/8])),multisample:makeMultisampleState(descriptor+136),fragment:makeFragmentState(Number((growMemViews(),HEAPU64)[(descriptor+160)/8]))};return desc},fillLimitStruct:(limits,limitsOutPtr)=>{var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsOutPtr/8]);function setLimitValueU32(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;(growMemViews(),HEAPU32)[(basePtr+limitOffset)/4]=limitValue}function setLimitValueU64(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;writeI53ToI64(basePtr+limitOffset,limitValue)}setLimitValueU32("maxTextureDimension1D",limitsOutPtr,8);setLimitValueU32("maxTextureDimension2D",limitsOutPtr,12);setLimitValueU32("maxTextureDimension3D",limitsOutPtr,16);setLimitValueU32("maxTextureArrayLayers",limitsOutPtr,20);setLimitValueU32("maxBindGroups",limitsOutPtr,24);setLimitValueU32("maxBindGroupsPlusVertexBuffers",limitsOutPtr,28);setLimitValueU32("maxBindingsPerBindGroup",limitsOutPtr,32);setLimitValueU32("maxDynamicUniformBuffersPerPipelineLayout",limitsOutPtr,36);setLimitValueU32("maxDynamicStorageBuffersPerPipelineLayout",limitsOutPtr,40);setLimitValueU32("maxSampledTexturesPerShaderStage",limitsOutPtr,44);setLimitValueU32("maxSamplersPerShaderStage",limitsOutPtr,48);setLimitValueU32("maxStorageBuffersPerShaderStage",limitsOutPtr,52);setLimitValueU32("maxStorageTexturesPerShaderStage",limitsOutPtr,56);setLimitValueU32("maxUniformBuffersPerShaderStage",limitsOutPtr,60);setLimitValueU32("minUniformBufferOffsetAlignment",limitsOutPtr,80);setLimitValueU32("minStorageBufferOffsetAlignment",limitsOutPtr,84);setLimitValueU64("maxUniformBufferBindingSize",limitsOutPtr,64);setLimitValueU64("maxStorageBufferBindingSize",limitsOutPtr,72);setLimitValueU32("maxVertexBuffers",limitsOutPtr,88);setLimitValueU64("maxBufferSize",limitsOutPtr,96);setLimitValueU32("maxVertexAttributes",limitsOutPtr,104);setLimitValueU32("maxVertexBufferArrayStride",limitsOutPtr,108);setLimitValueU32("maxInterStageShaderVariables",limitsOutPtr,112);setLimitValueU32("maxColorAttachments",limitsOutPtr,116);setLimitValueU32("maxColorAttachmentBytesPerSample",limitsOutPtr,120);setLimitValueU32("maxComputeWorkgroupStorageSize",limitsOutPtr,124);setLimitValueU32("maxComputeInvocationsPerWorkgroup",limitsOutPtr,128);setLimitValueU32("maxComputeWorkgroupSizeX",limitsOutPtr,132);setLimitValueU32("maxComputeWorkgroupSizeY",limitsOutPtr,136);setLimitValueU32("maxComputeWorkgroupSizeZ",limitsOutPtr,140);setLimitValueU32("maxComputeWorkgroupsPerDimension",limitsOutPtr,144);setLimitValueU32("maxImmediateSize",limitsOutPtr,148);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;setLimitValueU32("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20,limits.maxStorageTexturesPerShaderStage);setLimitValueU32("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28,limits.maxStorageTexturesPerShaderStage)}},fillAdapterInfoStruct:(info,infoStruct)=>{(growMemViews(),HEAPU32)[(infoStruct+88)/4]=info.subgroupMinSize;(growMemViews(),HEAPU32)[(infoStruct+92)/4]=info.subgroupMaxSize;var strs=info.vendor+info.architecture+info.device+info.description;var strPtr=stringToNewUTF8(strs);var vendorLen=lengthBytesUTF8(info.vendor);WebGPU.setStringView(infoStruct+8,strPtr,vendorLen);strPtr+=vendorLen;var architectureLen=lengthBytesUTF8(info.architecture);WebGPU.setStringView(infoStruct+24,strPtr,architectureLen);strPtr+=architectureLen;var deviceLen=lengthBytesUTF8(info.device);WebGPU.setStringView(infoStruct+40,strPtr,deviceLen);strPtr+=deviceLen;var descriptionLen=lengthBytesUTF8(info.description);WebGPU.setStringView(infoStruct+56,strPtr,descriptionLen);strPtr+=descriptionLen;(growMemViews(),HEAP32)[(infoStruct+72)/4]=2;var adapterType=info.isFallbackAdapter?3:4;(growMemViews(),HEAP32)[(infoStruct+76)/4]=adapterType;(growMemViews(),HEAPU32)[(infoStruct+80)/4]=0;(growMemViews(),HEAPU32)[(infoStruct+84)/4]=0},AddressMode:[,"clamp-to-edge","repeat","mirror-repeat"],BlendFactor:[,"zero","one","src","one-minus-src","src-alpha","one-minus-src-alpha","dst","one-minus-dst","dst-alpha","one-minus-dst-alpha","src-alpha-saturated","constant","one-minus-constant","src1","one-minus-src1","src1-alpha","one-minus-src1-alpha"],BlendOperation:[,"add","subtract","reverse-subtract","min","max"],BufferBindingType:[,,"uniform","storage","read-only-storage"],BufferMapState:[,"unmapped","pending","mapped"],CompareFunction:[,"never","less","equal","less-equal","greater","not-equal","greater-equal","always"],CompilationInfoRequestStatus:[,"success","callback-cancelled"],ComponentSwizzle:[,"0","1","r","g","b","a"],CompositeAlphaMode:[,"opaque","premultiplied","unpremultiplied","inherit"],CullMode:[,"none","front","back"],ErrorFilter:[,"validation","out-of-memory","internal"],FeatureLevel:[,"compatibility","core"],FeatureName:{1:"core-features-and-limits",2:"depth-clip-control",3:"depth32float-stencil8",4:"texture-compression-bc",5:"texture-compression-bc-sliced-3d",6:"texture-compression-etc2",7:"texture-compression-astc",8:"texture-compression-astc-sliced-3d",9:"timestamp-query",10:"indirect-first-instance",11:"shader-f16",12:"rg11b10ufloat-renderable",13:"bgra8unorm-storage",14:"float32-filterable",15:"float32-blendable",16:"clip-distances",17:"dual-source-blending",18:"subgroups",19:"texture-formats-tier1",20:"texture-formats-tier2",21:"primitive-index",22:"texture-component-swizzle",327692:"chromium-experimental-unorm16-texture-formats",327729:"chromium-experimental-multi-draw-indirect"},FilterMode:[,"nearest","linear"],FrontFace:[,"ccw","cw"],IndexFormat:[,"uint16","uint32"],InstanceFeatureName:[,"timed-wait-any","shader-source-spirv","multiple-devices-per-adapter"],LoadOp:[,"load","clear"],MipmapFilterMode:[,"nearest","linear"],OptionalBool:["false","true"],PowerPreference:[,"low-power","high-performance"],PredefinedColorSpace:[,"srgb","display-p3"],PrimitiveTopology:[,"point-list","line-list","line-strip","triangle-list","triangle-strip"],QueryType:[,"occlusion","timestamp"],SamplerBindingType:[,,"filtering","non-filtering","comparison"],Status:[,"success","error"],StencilOperation:[,"keep","zero","replace","invert","increment-clamp","decrement-clamp","increment-wrap","decrement-wrap"],StorageTextureAccess:[,,"write-only","read-only","read-write"],StoreOp:[,"store","discard"],SurfaceGetCurrentTextureStatus:[,"success-optimal","success-suboptimal","timeout","outdated","lost","error"],TextureAspect:[,"all","stencil-only","depth-only"],TextureDimension:[,"1d","2d","3d"],TextureFormat:[,"r8unorm","r8snorm","r8uint","r8sint","r16unorm","r16snorm","r16uint","r16sint","r16float","rg8unorm","rg8snorm","rg8uint","rg8sint","r32float","r32uint","r32sint","rg16unorm","rg16snorm","rg16uint","rg16sint","rg16float","rgba8unorm","rgba8unorm-srgb","rgba8snorm","rgba8uint","rgba8sint","bgra8unorm","bgra8unorm-srgb","rgb10a2uint","rgb10a2unorm","rg11b10ufloat","rgb9e5ufloat","rg32float","rg32uint","rg32sint","rgba16unorm","rgba16snorm","rgba16uint","rgba16sint","rgba16float","rgba32float","rgba32uint","rgba32sint","stencil8","depth16unorm","depth24plus","depth24plus-stencil8","depth32float","depth32float-stencil8","bc1-rgba-unorm","bc1-rgba-unorm-srgb","bc2-rgba-unorm","bc2-rgba-unorm-srgb","bc3-rgba-unorm","bc3-rgba-unorm-srgb","bc4-r-unorm","bc4-r-snorm","bc5-rg-unorm","bc5-rg-snorm","bc6h-rgb-ufloat","bc6h-rgb-float","bc7-rgba-unorm","bc7-rgba-unorm-srgb","etc2-rgb8unorm","etc2-rgb8unorm-srgb","etc2-rgb8a1unorm","etc2-rgb8a1unorm-srgb","etc2-rgba8unorm","etc2-rgba8unorm-srgb","eac-r11unorm","eac-r11snorm","eac-rg11unorm","eac-rg11snorm","astc-4x4-unorm","astc-4x4-unorm-srgb","astc-5x4-unorm","astc-5x4-unorm-srgb","astc-5x5-unorm","astc-5x5-unorm-srgb","astc-6x5-unorm","astc-6x5-unorm-srgb","astc-6x6-unorm","astc-6x6-unorm-srgb","astc-8x5-unorm","astc-8x5-unorm-srgb","astc-8x6-unorm","astc-8x6-unorm-srgb","astc-8x8-unorm","astc-8x8-unorm-srgb","astc-10x5-unorm","astc-10x5-unorm-srgb","astc-10x6-unorm","astc-10x6-unorm-srgb","astc-10x8-unorm","astc-10x8-unorm-srgb","astc-10x10-unorm","astc-10x10-unorm-srgb","astc-12x10-unorm","astc-12x10-unorm-srgb","astc-12x12-unorm","astc-12x12-unorm-srgb"],TextureSampleType:[,,"float","unfilterable-float","depth","sint","uint"],TextureViewDimension:[,"1d","2d","2d-array","cube","cube-array","3d"],ToneMappingMode:[,"standard","extended"],VertexFormat:[,"uint8","uint8x2","uint8x4","sint8","sint8x2","sint8x4","unorm8","unorm8x2","unorm8x4","snorm8","snorm8x2","snorm8x4","uint16","uint16x2","uint16x4","sint16","sint16x2","sint16x4","unorm16","unorm16x2","unorm16x4","snorm16","snorm16x2","snorm16x4","float16","float16x2","float16x4","float32","float32x2","float32x3","float32x4","uint32","uint32x2","uint32x3","uint32x4","sint32","sint32x2","sint32x3","sint32x4","unorm10-10-10-2","unorm8x4-bgra"],VertexStepMode:[,"vertex","instance"],WGSLLanguageFeatureName:[,"readonly_and_readwrite_storage_textures","packed_4x8_integer_dot_product","unrestricted_pointer_parameters","pointer_composite_access","uniform_buffer_standard_layout","subgroup_id","texture_and_sampler_let","subgroup_uniformity","texture_formats_tier1"]};var emwgpuStringToInt_DeviceLostReason={undefined:1,unknown:1,destroyed:2};var runtimeKeepalivePop=()=>{runtimeKeepaliveCounter-=1};function _emwgpuAdapterRequestDevice(adapterPtr,futureId,deviceLostFutureId,devicePtr,queuePtr,descriptor){adapterPtr=bigintToI53Checked(adapterPtr);futureId=bigintToI53Checked(futureId);deviceLostFutureId=bigintToI53Checked(deviceLostFutureId);devicePtr=bigintToI53Checked(devicePtr);queuePtr=bigintToI53Checked(queuePtr);descriptor=bigintToI53Checked(descriptor);var adapter=WebGPU.getJsObject(adapterPtr);var desc={};if(descriptor){var requiredFeatureCount=Number((growMemViews(),HEAPU64)[(descriptor+24)/8]);if(requiredFeatureCount){var requiredFeaturesPtr=Number((growMemViews(),HEAPU64)[(descriptor+32)/8]);desc["requiredFeatures"]=Array.from((growMemViews(),HEAPU32).subarray(requiredFeaturesPtr/4,(requiredFeaturesPtr+requiredFeatureCount*4)/4),feature=>WebGPU.FeatureName[feature])}var limitsPtr=Number((growMemViews(),HEAPU64)[(descriptor+40)/8]);if(limitsPtr){var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsPtr/8]);var requiredLimits={};function setLimitU32IfDefined(name,basePtr,limitOffset,ignoreIfZero=false){var ptr=basePtr+limitOffset;var value=(growMemViews(),HEAPU32)[ptr/4];if(value!=4294967295&&(!ignoreIfZero||value!=0)){requiredLimits[name]=value}}function setLimitU64IfDefined(name,basePtr,limitOffset){var ptr=basePtr+limitOffset;var limitPart1=(growMemViews(),HEAPU32)[ptr/4];var limitPart2=(growMemViews(),HEAPU32)[(ptr+4)/4];if(limitPart1!=4294967295||limitPart2!=4294967295){requiredLimits[name]=readI53FromI64(ptr)}}setLimitU32IfDefined("maxTextureDimension1D",limitsPtr,8);setLimitU32IfDefined("maxTextureDimension2D",limitsPtr,12);setLimitU32IfDefined("maxTextureDimension3D",limitsPtr,16);setLimitU32IfDefined("maxTextureArrayLayers",limitsPtr,20);setLimitU32IfDefined("maxBindGroups",limitsPtr,24);setLimitU32IfDefined("maxBindGroupsPlusVertexBuffers",limitsPtr,28);setLimitU32IfDefined("maxBindingsPerBindGroup",limitsPtr,32);setLimitU32IfDefined("maxDynamicUniformBuffersPerPipelineLayout",limitsPtr,36);setLimitU32IfDefined("maxDynamicStorageBuffersPerPipelineLayout",limitsPtr,40);setLimitU32IfDefined("maxSampledTexturesPerShaderStage",limitsPtr,44);setLimitU32IfDefined("maxSamplersPerShaderStage",limitsPtr,48);setLimitU32IfDefined("maxStorageBuffersPerShaderStage",limitsPtr,52);setLimitU32IfDefined("maxStorageTexturesPerShaderStage",limitsPtr,56);setLimitU32IfDefined("maxUniformBuffersPerShaderStage",limitsPtr,60);setLimitU32IfDefined("minUniformBufferOffsetAlignment",limitsPtr,80);setLimitU32IfDefined("minStorageBufferOffsetAlignment",limitsPtr,84);setLimitU64IfDefined("maxUniformBufferBindingSize",limitsPtr,64);setLimitU64IfDefined("maxStorageBufferBindingSize",limitsPtr,72);setLimitU32IfDefined("maxVertexBuffers",limitsPtr,88);setLimitU64IfDefined("maxBufferSize",limitsPtr,96);setLimitU32IfDefined("maxVertexAttributes",limitsPtr,104);setLimitU32IfDefined("maxVertexBufferArrayStride",limitsPtr,108);setLimitU32IfDefined("maxInterStageShaderVariables",limitsPtr,112);setLimitU32IfDefined("maxColorAttachments",limitsPtr,116);setLimitU32IfDefined("maxColorAttachmentBytesPerSample",limitsPtr,120);setLimitU32IfDefined("maxComputeWorkgroupStorageSize",limitsPtr,124);setLimitU32IfDefined("maxComputeInvocationsPerWorkgroup",limitsPtr,128);setLimitU32IfDefined("maxComputeWorkgroupSizeX",limitsPtr,132);setLimitU32IfDefined("maxComputeWorkgroupSizeY",limitsPtr,136);setLimitU32IfDefined("maxComputeWorkgroupSizeZ",limitsPtr,140);setLimitU32IfDefined("maxComputeWorkgroupsPerDimension",limitsPtr,144);setLimitU32IfDefined("maxImmediateSize",limitsPtr,148,true);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;if("maxStorageBuffersInVertexStage"in GPUSupportedLimits.prototype){setLimitU32IfDefined("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16);setLimitU32IfDefined("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20);setLimitU32IfDefined("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24);setLimitU32IfDefined("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28)}}desc["requiredLimits"]=requiredLimits}var defaultQueuePtr=Number((growMemViews(),HEAPU64)[(descriptor+48)/8]);if(defaultQueuePtr){var defaultQueueDesc={label:WebGPU.makeStringFromOptionalStringView(defaultQueuePtr+8)};desc["defaultQueue"]=defaultQueueDesc}desc["label"]=WebGPU.makeStringFromOptionalStringView(descriptor+8)}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,adapter.requestDevice(desc).then(device=>{runtimeKeepalivePop();callUserCallback(()=>{WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);devicePtr=BigInt(devicePtr);WebGPU.Internals.futureInsert(deviceLostFutureId,device.lost.then(info=>{callUserCallback(()=>{device.onuncapturederror=ev=>{};var sp=stackSave();var messagePtr=stringToUTF8OnStack(info.message);_emwgpuOnDeviceLostCompleted(deviceLostFutureId,emwgpuStringToInt_DeviceLostReason[info.reason],BigInt(messagePtr));stackRestore(sp)})}));device.onuncapturederror=ev=>{var type=5;if(ev.error instanceof GPUValidationError)type=2;else if(ev.error instanceof GPUOutOfMemoryError)type=3;else if(ev.error instanceof GPUInternalError)type=4;var sp=stackSave();var messagePtr=stringToUTF8OnStack(ev.error.message);_emwgpuOnUncapturedError(BigInt(devicePtr),type,BigInt(messagePtr));stackRestore(sp)};_emwgpuOnRequestDeviceCompleted(futureId,1,BigInt(devicePtr),0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestDeviceCompleted(futureId,3,BigInt(devicePtr),BigInt(messagePtr));if(deviceLostFutureId){_emwgpuOnDeviceLostCompleted(deviceLostFutureId,4,BigInt(messagePtr))}stackRestore(sp)})}))}function _emwgpuBufferDestroy(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(onUnmap){for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]}buffer.destroy()}var warnOnce=text=>{warnOnce.shown||={};if(!warnOnce.shown[text]){warnOnce.shown[text]=1;if(ENVIRONMENT_IS_NODE)text="warning: "+text;err(text)}};var _emwgpuBufferGetConstMappedRange=function(bufferPtr,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);if(size==-1)size=undefined;var mapped;try{mapped=buffer.getMappedRange(offset,size)}catch(ex){return 0n}var data=_memalign(16,mapped.byteLength);(growMemViews(),HEAPU8).set(new Uint8Array(mapped),data);WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(()=>_free(data));return data})();return BigInt(ret)};var _emwgpuBufferMapAsync=function(bufferPtr,futureId,mode,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);futureId=bigintToI53Checked(futureId);mode=bigintToI53Checked(mode);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var buffer=WebGPU.getJsObject(bufferPtr);WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[];if(size==-1)size=undefined;runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,buffer.mapAsync(mode,offset,size).then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnMapAsyncCompleted(futureId,1,0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);var status=ex.name==="AbortError"?4:ex.name==="OperationError"?3:0;_emwgpuOnMapAsyncCompleted(futureId,status,BigInt(messagePtr));delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]})}))};function _emwgpuBufferUnmap(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(!onUnmap){return}for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];buffer.unmap()}function _emwgpuDelete(ptr){ptr=bigintToI53Checked(ptr);delete WebGPU.Internals.jsObjects[ptr]}function _emwgpuDeviceCreateBuffer(devicePtr,descriptor,bufferPtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);bufferPtr=bigintToI53Checked(bufferPtr);var mappedAtCreation=!!(growMemViews(),HEAPU32)[(descriptor+40)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),usage:(growMemViews(),HEAPU32)[(descriptor+24)/4],size:readI53FromI64(descriptor+32),mappedAtCreation};var device=WebGPU.getJsObject(devicePtr);var buffer;try{buffer=device.createBuffer(desc)}catch(ex){return false}WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);if(mappedAtCreation){WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[]}return true}function _emwgpuDeviceCreateShaderModule(devicePtr,descriptor,shaderModulePtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);shaderModulePtr=bigintToI53Checked(shaderModulePtr);var nextInChainPtr=Number((growMemViews(),HEAPU64)[descriptor/8]);var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),code:""};switch(sType){case 2:{desc["code"]=WebGPU.makeStringFromStringView(nextInChainPtr+16);break}}var device=WebGPU.getJsObject(devicePtr);WebGPU.Internals.jsObjectInsert(shaderModulePtr,device.createShaderModule(desc))}var _emwgpuDeviceDestroy=devicePtr=>{const device=WebGPU.getJsObject(devicePtr);device.onuncapturederror=null;device.destroy()};function _emwgpuInstanceRequestAdapter(instancePtr,futureId,options,adapterPtr){instancePtr=bigintToI53Checked(instancePtr);futureId=bigintToI53Checked(futureId);options=bigintToI53Checked(options);adapterPtr=bigintToI53Checked(adapterPtr);var opts;if(options){opts={featureLevel:WebGPU.FeatureLevel[(growMemViews(),HEAP32)[(options+8)/4]],powerPreference:WebGPU.PowerPreference[(growMemViews(),HEAP32)[(options+12)/4]],forceFallbackAdapter:!!(growMemViews(),HEAPU32)[(options+16)/4]};var nextInChainPtr=Number((growMemViews(),HEAPU64)[options/8]);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var webxrOptions=nextInChainPtr;opts.xrCompatible=!!(growMemViews(),HEAPU32)[(webxrOptions+16)/4]}}if(!("gpu"in navigator)){var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (navigator.gpu is not available)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp);return}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,navigator.gpu.requestAdapter(opts).then(adapter=>{runtimeKeepalivePop();callUserCallback(()=>{if(adapter){WebGPU.Internals.jsObjectInsert(adapterPtr,adapter);_emwgpuOnRequestAdapterCompleted(futureId,1,BigInt(adapterPtr),0n)}else{var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (requestAdapter returned null)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)}})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestAdapterCompleted(futureId,4,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)})}))}var _emwgpuQueueOnSubmittedWorkDone=function(queuePtr,futureId){queuePtr=bigintToI53Checked(queuePtr);futureId=bigintToI53Checked(futureId);var queue=WebGPU.getJsObject(queuePtr);runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,queue.onSubmittedWorkDone().then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnWorkDoneCompleted(futureId,1)})}))};var _emwgpuWaitAny=function(futurePtr,futureCount,timeoutMSPtr){futurePtr=bigintToI53Checked(futurePtr);futureCount=bigintToI53Checked(futureCount);timeoutMSPtr=bigintToI53Checked(timeoutMSPtr);return Asyncify.handleAsync(async()=>{var promises=[];if(timeoutMSPtr){var timeoutMS=(growMemViews(),HEAP32)[timeoutMSPtr/4];promises.length=futureCount+1;promises[futureCount]=new Promise(resolve=>setTimeout(resolve,timeoutMS,0))}else{promises.length=futureCount}for(var i=0;i<futureCount;++i){var futureId=readI53FromI64(futurePtr+i*8);if(!(futureId in WebGPU.Internals.futures)){return futureId}promises[i]=WebGPU.Internals.futures[futureId]}const firstResolvedFuture=await Promise.race(promises);delete WebGPU.Internals.futures[firstResolvedFuture];return firstResolvedFuture})};_emwgpuWaitAny.isAsync=true;var ENV={};var getExecutableName=()=>thisProgram||"./this.program";var getEnvStrings=()=>{if(!getEnvStrings.strings){var lang=(typeof navigator=="object"&&navigator.language||"C").replace("-","_")+".UTF-8";var env={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:lang,_:getExecutableName()};for(var x in ENV){if(ENV[x]===undefined)delete env[x];else env[x]=ENV[x]}var strings=[];for(var x in env){strings.push(`${x}=${env[x]}`)}getEnvStrings.strings=strings}return getEnvStrings.strings};function _environ_get(__environ,environ_buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(14,0,1,__environ,environ_buf);__environ=bigintToI53Checked(__environ);environ_buf=bigintToI53Checked(environ_buf);var bufSize=0;var envp=0;for(var string of getEnvStrings()){var ptr=environ_buf+bufSize;(growMemViews(),HEAPU64)[(__environ+envp)/8]=BigInt(ptr);bufSize+=stringToUTF8(string,ptr,Infinity)+1;envp+=8}return 0}function _environ_sizes_get(penviron_count,penviron_buf_size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(15,0,1,penviron_count,penviron_buf_size);penviron_count=bigintToI53Checked(penviron_count);penviron_buf_size=bigintToI53Checked(penviron_buf_size);var strings=getEnvStrings();(growMemViews(),HEAPU64)[penviron_count/8]=BigInt(strings.length);var bufSize=0;for(var string of strings){bufSize+=lengthBytesUTF8(string)+1}(growMemViews(),HEAPU64)[penviron_buf_size/8]=BigInt(bufSize);return 0}function _fd_close(fd){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(16,0,1,fd);try{var stream=SYSCALLS.getStreamFromFD(fd);FS.close(stream);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doReadv=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.read(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len)break;if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_read(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(17,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doReadv(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _fd_seek(fd,offset,whence,newOffset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(18,0,1,fd,offset,whence,newOffset);offset=bigintToI53Checked(offset);newOffset=bigintToI53Checked(newOffset);try{if(isNaN(offset))return 61;var stream=SYSCALLS.getStreamFromFD(fd);FS.llseek(stream,offset,whence);(growMemViews(),HEAP64)[newOffset/8]=BigInt(stream.position);if(stream.getdents&&offset===0&&whence===0)stream.getdents=null;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doWritev=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.write(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len){break}if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_write(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(19,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _random_get(buffer,size){buffer=bigintToI53Checked(buffer);size=bigintToI53Checked(size);try{randomFill((growMemViews(),HEAPU8).subarray(buffer,buffer+size));return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _wgpuAdapterGetInfo(adapterPtr,info){adapterPtr=bigintToI53Checked(adapterPtr);info=bigintToI53Checked(info);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillAdapterInfoStruct(adapter.info,info);return 1}function _wgpuAdapterGetLimits(adapterPtr,limitsOutPtr){adapterPtr=bigintToI53Checked(adapterPtr);limitsOutPtr=bigintToI53Checked(limitsOutPtr);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillLimitStruct(adapter.limits,limitsOutPtr);return 1}function _wgpuAdapterHasFeature(adapterPtr,featureEnumValue){adapterPtr=bigintToI53Checked(adapterPtr);var adapter=WebGPU.getJsObject(adapterPtr);return adapter.features.has(WebGPU.FeatureName[featureEnumValue])}var _wgpuBufferGetSize=function(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);return buffer.size})();return BigInt(ret)};var _wgpuCommandEncoderBeginComputePass=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),timestampWrites:WebGPU.makePassTimestampWrites(Number((growMemViews(),HEAPU64)[(descriptor+24)/8]))}}var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateComputePassEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.beginComputePass(desc));return ptr})();return BigInt(ret)};function _wgpuCommandEncoderCopyBufferToBuffer(encoderPtr,srcPtr,srcOffset,dstPtr,dstOffset,size){encoderPtr=bigintToI53Checked(encoderPtr);srcPtr=bigintToI53Checked(srcPtr);srcOffset=bigintToI53Checked(srcOffset);dstPtr=bigintToI53Checked(dstPtr);dstOffset=bigintToI53Checked(dstOffset);size=bigintToI53Checked(size);var commandEncoder=WebGPU.getJsObject(encoderPtr);var src=WebGPU.getJsObject(srcPtr);var dst=WebGPU.getJsObject(dstPtr);commandEncoder.copyBufferToBuffer(src,srcOffset,dst,dstOffset,size)}var _wgpuCommandEncoderFinish=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateCommandBuffer(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.finish());return ptr})();return BigInt(ret)};function _wgpuComputePassEncoderDispatchWorkgroups(passPtr,x,y,z){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.dispatchWorkgroups(x,y,z)}function _wgpuComputePassEncoderEnd(passPtr){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.end()}function _wgpuComputePassEncoderSetBindGroup(passPtr,groupIndex,groupPtr,dynamicOffsetCount,dynamicOffsetsPtr){passPtr=bigintToI53Checked(passPtr);groupPtr=bigintToI53Checked(groupPtr);dynamicOffsetCount=bigintToI53Checked(dynamicOffsetCount);dynamicOffsetsPtr=bigintToI53Checked(dynamicOffsetsPtr);var pass=WebGPU.getJsObject(passPtr);var group=WebGPU.getJsObject(groupPtr);if(dynamicOffsetCount==0){pass.setBindGroup(groupIndex,group)}else{pass.setBindGroup(groupIndex,group,(growMemViews(),HEAPU32),dynamicOffsetsPtr/4,dynamicOffsetCount)}}function _wgpuComputePassEncoderSetPipeline(passPtr,pipelinePtr){passPtr=bigintToI53Checked(passPtr);pipelinePtr=bigintToI53Checked(pipelinePtr);var pass=WebGPU.getJsObject(passPtr);var pipeline=WebGPU.getJsObject(pipelinePtr);pass.setPipeline(pipeline)}var _wgpuComputePipelineGetBindGroupLayout=function(pipelinePtr,groupIndex){pipelinePtr=bigintToI53Checked(pipelinePtr);var ret=(()=>{var pipeline=WebGPU.getJsObject(pipelinePtr);var ptr=_emwgpuCreateBindGroupLayout(0n);WebGPU.Internals.jsObjectInsert(ptr,pipeline.getBindGroupLayout(groupIndex));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateBindGroup=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{function makeEntry(entryPtr){var bufferPtr=Number((growMemViews(),HEAPU64)[(entryPtr+16)/8]);var samplerPtr=Number((growMemViews(),HEAPU64)[(entryPtr+40)/8]);var textureViewPtr=Number((growMemViews(),HEAPU64)[(entryPtr+48)/8]);var externalTexturePtr=0;WebGPU.iterateExtensions(entryPtr,{327681:ptr=>{externalTexturePtr=Number((growMemViews(),HEAPU64)[(ptr+16)/8])}});var resource;if(bufferPtr){var size=readI53FromI64(entryPtr+32);if(size==-1)size=undefined;resource={buffer:WebGPU.getJsObject(bufferPtr),offset:readI53FromI64(entryPtr+24),size}}else{resource=WebGPU.getJsObject(samplerPtr||textureViewPtr||externalTexturePtr)}return{binding:(growMemViews(),HEAPU32)[(entryPtr+8)/4],resource}}function makeEntries(count,entriesPtrs){var entries=[];for(var i=0;i<count;++i){entries.push(makeEntry(entriesPtrs+56*i))}return entries}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),entries:makeEntries(Number((growMemViews(),HEAPU64)[(descriptor+32)/8]),Number((growMemViews(),HEAPU64)[(descriptor+40)/8]))};var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateBindGroup(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createBindGroup(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateCommandEncoder=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8)}}var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateCommandEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createCommandEncoder(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateComputePipeline=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc=WebGPU.makeComputePipelineDesc(descriptor);var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateComputePipeline(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createComputePipeline(desc));return ptr})();return BigInt(ret)};function _wgpuInstanceHasWGSLLanguageFeature(instance,featureEnumValue){instance=bigintToI53Checked(instance);if(!("wgslLanguageFeatures"in navigator.gpu)){return false}return navigator.gpu.wgslLanguageFeatures.has(WebGPU.WGSLLanguageFeatureName[featureEnumValue])}var _wgpuQueueSubmit=function(queuePtr,commandCount,commands){queuePtr=bigintToI53Checked(queuePtr);commandCount=bigintToI53Checked(commandCount);commands=bigintToI53Checked(commands);var queue=WebGPU.getJsObject(queuePtr);var cmds=Array.from((growMemViews(),HEAP64).subarray(commands/8,(commands+commandCount*8)/8),id=>WebGPU.getJsObject(id));queue.submit(cmds)};function _wgpuQueueWriteBuffer(queuePtr,bufferPtr,bufferOffset,data,size){queuePtr=bigintToI53Checked(queuePtr);bufferPtr=bigintToI53Checked(bufferPtr);bufferOffset=bigintToI53Checked(bufferOffset);data=bigintToI53Checked(data);size=bigintToI53Checked(size);var queue=WebGPU.getJsObject(queuePtr);var buffer=WebGPU.getJsObject(bufferPtr);var subarray=(growMemViews(),HEAPU8).subarray(data,data+size);queue.writeBuffer(buffer,bufferOffset,subarray,0,size)}var Asyncify={instrumentWasmImports(imports){var importPattern=/^(invoke_.*|__asyncjs__.*)$/;for(let[x,original]of Object.entries(imports)){if(typeof original=="function"){let isAsyncifyImport=original.isAsync||importPattern.test(x);if(isAsyncifyImport){imports[x]=original=new WebAssembly.Suspending(original)}}}},instrumentFunction(original){var wrapper=(...args)=>original(...args);return wrapper},instrumentWasmExports(exports){var exportPattern=/^(wllama_start|wllama_action|main|__main_argc_argv)$/;Asyncify.asyncExports=new Set;var ret={};for(let[x,original]of Object.entries(exports)){if(typeof original=="function"){let isAsyncifyExport=exportPattern.test(x);if(isAsyncifyExport){Asyncify.asyncExports.add(original);original=Asyncify.makeAsyncFunction(original)}var wrapper=Asyncify.instrumentFunction(original);ret[x]=wrapper}else{ret[x]=original}}return ret},asyncExports:null,isAsyncExport(func){return Asyncify.asyncExports?.has(func)},handleAsync:async startAsync=>{runtimeKeepalivePush();try{return await startAsync()}finally{runtimeKeepalivePop()}},handleSleep:startAsync=>Asyncify.handleAsync(()=>new Promise(startAsync)),makeAsyncFunction(original){return WebAssembly.promising(original)}};var getCFunc=ident=>{var func=Module["_"+ident];return func};var writeArrayToMemory=(array,buffer)=>{(growMemViews(),HEAP8).set(array,buffer)};var ccall=(ident,returnType,argTypes,args,opts)=>{var toC={pointer:p=>BigInt(p),string:str=>{var ret=0;if(str!==null&&str!==undefined&&str!==0){ret=stringToUTF8OnStack(str)}return BigInt(ret)},array:arr=>{var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return BigInt(ret)}};function convertReturnValue(ret){if(returnType==="string"){return UTF8ToString(Number(ret))}if(returnType==="pointer")return Number(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i])}else{cArgs[i]=args[i]}}}var ret=func(...cArgs);function onDone(ret){if(stack!==0)stackRestore(stack);return convertReturnValue(ret)}var asyncMode=opts?.async;if(asyncMode)return ret.then(onDone);ret=onDone(ret);return ret};var cwrap=(ident,returnType,argTypes,opts)=>{var numericArgs=!argTypes||argTypes.every(type=>type==="number"||type==="boolean");var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return(...args)=>ccall(ident,returnType,argTypes,args,opts)};var FS_createPath=(...args)=>FS.createPath(...args);var FS_unlink=(...args)=>FS.unlink(...args);var FS_createLazyFile=(...args)=>FS.createLazyFile(...args);var FS_createDevice=(...args)=>FS.createDevice(...args);PThread.init();FS.createPreloadedFile=FS_createPreloadedFile;FS.preloadFile=FS_preloadFile;FS.staticInit();{initMemory();if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(Module["preloadPlugins"])preloadPlugins=Module["preloadPlugins"];if(Module["print"])out=Module["print"];if(Module["printErr"])err=Module["printErr"];if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].shift()()}}}Module["ENV"]=ENV;Module["mmapAlloc"]=mmapAlloc;Module["wasmMemory"]=wasmMemory;Module["addRunDependency"]=addRunDependency;Module["removeRunDependency"]=removeRunDependency;Module["ccall"]=ccall;Module["cwrap"]=cwrap;Module["FS_preloadFile"]=FS_preloadFile;Module["FS_unlink"]=FS_unlink;Module["FS_createPath"]=FS_createPath;Module["FS_createDevice"]=FS_createDevice;Module["FS"]=FS;Module["FS_createDataFile"]=FS_createDataFile;Module["FS_createLazyFile"]=FS_createLazyFile;Module["MEMFS"]=MEMFS;var proxiedFunctionTable=[_proc_exit,exitOnMainThread,pthreadCreateProxied,___syscall_fcntl64,___syscall_fstat64,___syscall_getcwd,___syscall_getdents64,___syscall_ioctl,___syscall_lstat64,___syscall_newfstatat,___syscall_openat,___syscall_stat64,__mmap_js,__munmap_js,_environ_get,_environ_sizes_get,_fd_close,_fd_read,_fd_seek,_fd_write];function __asyncjs__js_file_read(path_ptr,offset,req_size,out_ptr){return Asyncify.handleAsync(async()=>await _wllama_js_file_read(UTF8ToString(Number(path_ptr)),Number(offset),Number(req_size),Number(out_ptr)))}__asyncjs__js_file_read.sig="jjjjj";var _wllama_malloc,_wllama_start,_wllama_action,_wllama_exit,_wllama_debug,_main,_malloc,_free,_emwgpuCreateBindGroup,_emwgpuCreateBindGroupLayout,_emwgpuCreateCommandBuffer,_emwgpuCreateCommandEncoder,_emwgpuCreateComputePassEncoder,_emwgpuCreateComputePipeline,_emwgpuCreateExternalTexture,_emwgpuCreatePipelineLayout,_emwgpuCreateQuerySet,_emwgpuCreateRenderBundle,_emwgpuCreateRenderBundleEncoder,_emwgpuCreateRenderPassEncoder,_emwgpuCreateRenderPipeline,_emwgpuCreateSampler,_emwgpuCreateSurface,_emwgpuCreateTexture,_emwgpuCreateTextureView,_emwgpuCreateAdapter,_emwgpuCreateBuffer,_emwgpuCreateDevice,_emwgpuCreateQueue,_emwgpuCreateShaderModule,_emwgpuOnDeviceLostCompleted,_emwgpuOnMapAsyncCompleted,_emwgpuOnRequestAdapterCompleted,_emwgpuOnRequestDeviceCompleted,_emwgpuOnWorkDoneCompleted,_emwgpuOnUncapturedError,__emscripten_tls_init,_pthread_self,_emscripten_builtin_memalign,__emscripten_thread_init,__emscripten_thread_crashed,__emscripten_run_js_on_main_thread,__emscripten_thread_free_data,__emscripten_thread_exit,__emscripten_check_mailbox,_memalign,___trap,_emscripten_stack_set_limits,__emscripten_stack_restore,__emscripten_stack_alloc,_emscripten_stack_get_current,__indirect_function_table,wasmTable;function assignWasmExports(wasmExports){_wllama_malloc=Module["_wllama_malloc"]=wasmExports["wllama_malloc"];_wllama_start=Module["_wllama_start"]=wasmExports["wllama_start"];_wllama_action=Module["_wllama_action"]=wasmExports["wllama_action"];_wllama_exit=Module["_wllama_exit"]=wasmExports["wllama_exit"];_wllama_debug=Module["_wllama_debug"]=wasmExports["wllama_debug"];_main=Module["_main"]=wasmExports["main"];_malloc=wasmExports["malloc"];_free=wasmExports["free"];_emwgpuCreateBindGroup=wasmExports["emwgpuCreateBindGroup"];_emwgpuCreateBindGroupLayout=wasmExports["emwgpuCreateBindGroupLayout"];_emwgpuCreateCommandBuffer=wasmExports["emwgpuCreateCommandBuffer"];_emwgpuCreateCommandEncoder=wasmExports["emwgpuCreateCommandEncoder"];_emwgpuCreateComputePassEncoder=wasmExports["emwgpuCreateComputePassEncoder"];_emwgpuCreateComputePipeline=wasmExports["emwgpuCreateComputePipeline"];_emwgpuCreateExternalTexture=wasmExports["emwgpuCreateExternalTexture"];_emwgpuCreatePipelineLayout=wasmExports["emwgpuCreatePipelineLayout"];_emwgpuCreateQuerySet=wasmExports["emwgpuCreateQuerySet"];_emwgpuCreateRenderBundle=wasmExports["emwgpuCreateRenderBundle"];_emwgpuCreateRenderBundleEncoder=wasmExports["emwgpuCreateRenderBundleEncoder"];_emwgpuCreateRenderPassEncoder=wasmExports["emwgpuCreateRenderPassEncoder"];_emwgpuCreateRenderPipeline=wasmExports["emwgpuCreateRenderPipeline"];_emwgpuCreateSampler=wasmExports["emwgpuCreateSampler"];_emwgpuCreateSurface=wasmExports["emwgpuCreateSurface"];_emwgpuCreateTexture=wasmExports["emwgpuCreateTexture"];_emwgpuCreateTextureView=wasmExports["emwgpuCreateTextureView"];_emwgpuCreateAdapter=wasmExports["emwgpuCreateAdapter"];_emwgpuCreateBuffer=wasmExports["emwgpuCreateBuffer"];_emwgpuCreateDevice=wasmExports["emwgpuCreateDevice"];_emwgpuCreateQueue=wasmExports["emwgpuCreateQueue"];_emwgpuCreateShaderModule=wasmExports["emwgpuCreateShaderModule"];_emwgpuOnDeviceLostCompleted=wasmExports["emwgpuOnDeviceLostCompleted"];_emwgpuOnMapAsyncCompleted=wasmExports["emwgpuOnMapAsyncCompleted"];_emwgpuOnRequestAdapterCompleted=wasmExports["emwgpuOnRequestAdapterCompleted"];_emwgpuOnRequestDeviceCompleted=wasmExports["emwgpuOnRequestDeviceCompleted"];_emwgpuOnWorkDoneCompleted=wasmExports["emwgpuOnWorkDoneCompleted"];_emwgpuOnUncapturedError=wasmExports["emwgpuOnUncapturedError"];__emscripten_tls_init=wasmExports["_emscripten_tls_init"];_pthread_self=wasmExports["pthread_self"];_emscripten_builtin_memalign=wasmExports["emscripten_builtin_memalign"];__emscripten_thread_init=wasmExports["_emscripten_thread_init"];__emscripten_thread_crashed=wasmExports["_emscripten_thread_crashed"];__emscripten_run_js_on_main_thread=wasmExports["_emscripten_run_js_on_main_thread"];__emscripten_thread_free_data=wasmExports["_emscripten_thread_free_data"];__emscripten_thread_exit=wasmExports["_emscripten_thread_exit"];__emscripten_check_mailbox=wasmExports["_emscripten_check_mailbox"];_memalign=wasmExports["memalign"];___trap=wasmExports["__trap"];_emscripten_stack_set_limits=wasmExports["emscripten_stack_set_limits"];__emscripten_stack_restore=wasmExports["_emscripten_stack_restore"];__emscripten_stack_alloc=wasmExports["_emscripten_stack_alloc"];_emscripten_stack_get_current=wasmExports["emscripten_stack_get_current"];__indirect_function_table=wasmTable=wasmExports["__indirect_function_table"]}var wasmImports;function assignWasmImports(){wasmImports={__asyncjs__js_file_read,__pthread_create_js:___pthread_create_js,__syscall_fcntl64:___syscall_fcntl64,__syscall_getcwd:___syscall_getcwd,__syscall_getdents64:___syscall_getdents64,__syscall_ioctl:___syscall_ioctl,__syscall_openat:___syscall_openat,__syscall_stat64:___syscall_stat64,_abort_js:__abort_js,_emscripten_init_main_thread_js:__emscripten_init_main_thread_js,_emscripten_notify_mailbox_postmessage:__emscripten_notify_mailbox_postmessage,_emscripten_receive_on_main_thread_js:__emscripten_receive_on_main_thread_js,_emscripten_thread_cleanup:__emscripten_thread_cleanup,_emscripten_thread_mailbox_await:__emscripten_thread_mailbox_await,_emscripten_thread_set_strongref:__emscripten_thread_set_strongref,_localtime_js:__localtime_js,_mmap_js:__mmap_js,_munmap_js:__munmap_js,_tzset_js:__tzset_js,clock_time_get:_clock_time_get,emscripten_check_blocking_allowed:_emscripten_check_blocking_allowed,emscripten_date_now:_emscripten_date_now,emscripten_exit_with_live_runtime:_emscripten_exit_with_live_runtime,emscripten_get_callstack:_emscripten_get_callstack,emscripten_get_heap_max:_emscripten_get_heap_max,emscripten_get_now:_emscripten_get_now,emscripten_has_asyncify:_emscripten_has_asyncify,emscripten_num_logical_cores:_emscripten_num_logical_cores,emscripten_resize_heap:_emscripten_resize_heap,emwgpuAdapterRequestDevice:_emwgpuAdapterRequestDevice,emwgpuBufferDestroy:_emwgpuBufferDestroy,emwgpuBufferGetConstMappedRange:_emwgpuBufferGetConstMappedRange,emwgpuBufferMapAsync:_emwgpuBufferMapAsync,emwgpuBufferUnmap:_emwgpuBufferUnmap,emwgpuDelete:_emwgpuDelete,emwgpuDeviceCreateBuffer:_emwgpuDeviceCreateBuffer,emwgpuDeviceCreateShaderModule:_emwgpuDeviceCreateShaderModule,emwgpuDeviceDestroy:_emwgpuDeviceDestroy,emwgpuInstanceRequestAdapter:_emwgpuInstanceRequestAdapter,emwgpuQueueOnSubmittedWorkDone:_emwgpuQueueOnSubmittedWorkDone,emwgpuWaitAny:_emwgpuWaitAny,environ_get:_environ_get,environ_sizes_get:_environ_sizes_get,exit:_exit,fd_close:_fd_close,fd_read:_fd_read,fd_seek:_fd_seek,fd_write:_fd_write,memory:wasmMemory,random_get:_random_get,wgpuAdapterGetInfo:_wgpuAdapterGetInfo,wgpuAdapterGetLimits:_wgpuAdapterGetLimits,wgpuAdapterHasFeature:_wgpuAdapterHasFeature,wgpuBufferGetSize:_wgpuBufferGetSize,wgpuCommandEncoderBeginComputePass:_wgpuCommandEncoderBeginComputePass,wgpuCommandEncoderCopyBufferToBuffer:_wgpuCommandEncoderCopyBufferToBuffer,wgpuCommandEncoderFinish:_wgpuCommandEncoderFinish,wgpuComputePassEncoderDispatchWorkgroups:_wgpuComputePassEncoderDispatchWorkgroups,wgpuComputePassEncoderEnd:_wgpuComputePassEncoderEnd,wgpuComputePassEncoderSetBindGroup:_wgpuComputePassEncoderSetBindGroup,wgpuComputePassEncoderSetPipeline:_wgpuComputePassEncoderSetPipeline,wgpuComputePipelineGetBindGroupLayout:_wgpuComputePipelineGetBindGroupLayout,wgpuDeviceCreateBindGroup:_wgpuDeviceCreateBindGroup,wgpuDeviceCreateCommandEncoder:_wgpuDeviceCreateCommandEncoder,wgpuDeviceCreateComputePipeline:_wgpuDeviceCreateComputePipeline,wgpuInstanceHasWGSLLanguageFeature:_wgpuInstanceHasWGSLLanguageFeature,wgpuQueueSubmit:_wgpuQueueSubmit,wgpuQueueWriteBuffer:_wgpuQueueWriteBuffer}}function applySignatureConversions(wasmExports){wasmExports=Object.assign({},wasmExports);var makeWrapper___PP=f=>(a0,a1,a2)=>f(a0,BigInt(a1?a1:0),BigInt(a2?a2:0));var makeWrapper_pp=f=>a0=>Number(f(BigInt(a0)));var makeWrapper__p=f=>a0=>f(BigInt(a0));var makeWrapper_p=f=>()=>Number(f());var makeWrapper_ppp=f=>(a0,a1)=>Number(f(BigInt(a0),BigInt(a1)));var makeWrapper__p_____=f=>(a0,a1,a2,a3,a4,a5)=>f(BigInt(a0),a1,a2,a3,a4,a5);var makeWrapper___p_p_=f=>(a0,a1,a2,a3,a4)=>f(a0,BigInt(a1),a2,BigInt(a3),a4);var makeWrapper__pp=f=>(a0,a1)=>f(BigInt(a0),BigInt(a1));wasmExports["main"]=makeWrapper___PP(wasmExports["main"]);wasmExports["malloc"]=makeWrapper_pp(wasmExports["malloc"]);wasmExports["free"]=makeWrapper__p(wasmExports["free"]);wasmExports["pthread_self"]=makeWrapper_p(wasmExports["pthread_self"]);wasmExports["emscripten_builtin_memalign"]=makeWrapper_ppp(wasmExports["emscripten_builtin_memalign"]);wasmExports["_emscripten_thread_init"]=makeWrapper__p_____(wasmExports["_emscripten_thread_init"]);wasmExports["_emscripten_run_js_on_main_thread"]=makeWrapper___p_p_(wasmExports["_emscripten_run_js_on_main_thread"]);wasmExports["_emscripten_thread_free_data"]=makeWrapper__p(wasmExports["_emscripten_thread_free_data"]);wasmExports["_emscripten_thread_exit"]=makeWrapper__p(wasmExports["_emscripten_thread_exit"]);wasmExports["memalign"]=makeWrapper_ppp(wasmExports["memalign"]);wasmExports["emscripten_stack_set_limits"]=makeWrapper__pp(wasmExports["emscripten_stack_set_limits"]);wasmExports["_emscripten_stack_restore"]=makeWrapper__p(wasmExports["_emscripten_stack_restore"]);wasmExports["_emscripten_stack_alloc"]=makeWrapper_pp(wasmExports["_emscripten_stack_alloc"]);wasmExports["emscripten_stack_get_current"]=makeWrapper_p(wasmExports["emscripten_stack_get_current"]);return wasmExports}async function callMain(){var entryFunction=_main;var argc=0;var argv=0;try{var ret=entryFunction(argc,BigInt(argv));ret=await ret;exitJS(ret,true);return ret}catch(e){return handleException(e)}}function run(){if(runDependencies>0){dependenciesFulfilled=run;return}if(ENVIRONMENT_IS_PTHREAD){initRuntime();return}preRun();if(runDependencies>0){dependenciesFulfilled=run;return}async function doRun(){Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();Module["onRuntimeInitialized"]?.();var noInitialRun=Module["noInitialRun"]||false;if(!noInitialRun)await callMain();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun()},1)}else{doRun()}}var wasmExports;if(!ENVIRONMENT_IS_PTHREAD){createWasm();run()}\n';

// src/worker.ts
var FILE_READ_REQ_EVENT = "fs.read_req";
var JSPI_STUB = `
if (!WebAssembly.Suspending) {
  // JSPI not available - stubs that keep the import/export tables valid.
  // Suspending wraps imports: identity is fine since async imports won't be called.
  WebAssembly.Suspending = function (fn) {
    // console.log(fn.toString());
    return fn;
  };
  // promising wraps exports: must return a Promise so ccall's ret.then() works.
  WebAssembly.promising = function (fn) {
    return function (...args) {
      try {
        return Promise.resolve(fn(...args));
      } catch (e) {
        return Promise.reject(e);
      }
    };
  };
}
`;
var ProxyToWorker = class {
  // filename -> Blob for async reads
  constructor(resources, nbThread, suppressNativeLog, logger) {
    __publicField(this, "resources");
    __publicField(this, "logger");
    __publicField(this, "suppressNativeLog");
    __publicField(this, "taskQueue", []);
    __publicField(this, "taskId", 1);
    __publicField(this, "resultQueue", []);
    __publicField(this, "busy", false);
    // is the work loop is running?
    __publicField(this, "worker");
    __publicField(this, "multiThread");
    __publicField(this, "nbThread");
    __publicField(this, "useAsyncFile");
    __publicField(this, "fileBlobs", /* @__PURE__ */ new Map());
    this.resources = resources;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 0;
    this.logger = logger;
    this.suppressNativeLog = suppressNativeLog;
    this.useAsyncFile = canUseAsyncFileRead(resources.compat);
  }
  getModuleCode() {
    return __async(this, null, function* () {
      if (!this.resources.jsPath) {
        if (this.resources.compat) {
          throw new Error(
            "compat mode is enabled but no jsPath was provided. Pass a worker JS via setCompat() or install @wllama/wllama-compat."
          );
        }
        return WLLAMA_EMSCRIPTEN_CODE;
      } else if (this.resources.jsPath.code) {
        return this.resources.jsPath.code;
      } else if (isString(this.resources.jsPath)) {
        const response = yield fetch(this.resources.jsPath);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch worker code from ${this.resources.jsPath}`
          );
        }
        return yield response.text();
      } else {
        throw new Error("No JS code provided for worker");
      }
    });
  }
  moduleInit(ggufFiles) {
    return __async(this, null, function* () {
      let moduleCode = JSPI_STUB + (yield this.getModuleCode());
      if (this.resources.noWebGPU) {
        moduleCode = 'try{Object.defineProperty(WorkerNavigator.prototype,"gpu",{get:()=>({requestAdapter:async()=>null})});}catch(e){}' + moduleCode;
      }
      let mainModuleCode = moduleCode.replace("var Module", "var ___Module");
      const runOptions = {
        pathConfig: {
          "wllama.wasm": this.resources.wasmPath
        },
        nbThread: this.nbThread,
        compat: this.resources.compat
      };
      const completeCode = [
        `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
        `function wModuleInit() { ${mainModuleCode}; return Module; }`,
        LLAMA_CPP_WORKER_CODE
      ].join(";\n\n");
      this.worker = createWorker(completeCode);
      this.worker.onmessage = this.onRecvMsg.bind(this);
      this.worker.onerror = this.logger.error;
      const res = yield this.pushTask({
        verb: "module.init",
        args: [
          new Blob([moduleCode], { type: "text/javascript" }),
          this.useAsyncFile
        ],
        callbackId: this.taskId++
      });
      const nativeFiles = [];
      for (const file of ggufFiles) {
        const needAllocBuffer = !this.useAsyncFile;
        const id = yield this.fileAlloc(
          file.name,
          file.blob.size,
          needAllocBuffer
        );
        nativeFiles.push(__spreadValues({ id }, file));
        if (this.useAsyncFile) {
          this.fileBlobs.set(file.name, file.blob);
        }
      }
      if (!this.useAsyncFile) {
        yield Promise.all(
          nativeFiles.map((file) => {
            return this.fileWrite(file.id, file.blob);
          })
        );
      }
      return res;
    });
  }
  wllamaStart() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.start",
        args: [],
        callbackId: this.taskId++
      });
      const parsedResult = this.parseResult(result);
      return parsedResult;
    });
  }
  wllamaAction(name, body) {
    return __async(this, null, function* () {
      const encodedMsg = glueSerialize(body);
      const result = yield this.pushTask({
        verb: "wllama.action",
        args: [name, encodedMsg],
        callbackId: this.taskId++
      });
      const parsedResult = glueDeserialize(result);
      if (parsedResult._name === "erro_evt") {
        throw new WllamaRuntimeError(parsedResult.message, "");
      }
      return parsedResult;
    });
  }
  /** Take a slot save file out of the worker's in-memory FS (the file is removed) */
  slotFileRead(fileName) {
    return __async(this, null, function* () {
      return yield this.pushTask({
        verb: "fs.slot_read",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  /** Put a slot save file into the worker's in-memory FS */
  slotFileWrite(fileName, data) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_write",
        args: [fileName, data],
        callbackId: this.taskId++
      });
    });
  }
  slotFileRemove(fileName) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_remove",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  wllamaExit() {
    return __async(this, null, function* () {
      if (this.worker) {
        this.worker.terminate();
      }
    });
  }
  wllamaDebug() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.debug",
        args: [],
        callbackId: this.taskId++
      });
      return JSON.parse(result);
    });
  }
  ///////////////////////////////////////
  /**
   * Allocate a new file in heapfs
   * @returns fileId, to be used by fileWrite()
   */
  fileAlloc(fileName, size, allocBuffer) {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "fs.alloc",
        args: [fileName, size, allocBuffer],
        callbackId: this.taskId++
      });
      return result.fileId;
    });
  }
  /**
   * Write a Blob to heapfs
   */
  fileWrite(fileId, blob) {
    return __async(this, null, function* () {
      const reader = blob.stream().getReader();
      let offset = 0;
      while (true) {
        const { done, value } = yield reader.read();
        if (done) break;
        const size = value.byteLength;
        yield this.pushTask(
          {
            verb: "fs.write",
            args: [fileId, value, offset],
            callbackId: this.taskId++
          },
          // @ts-ignore Type 'ArrayBufferLike' is not assignable to type 'ArrayBuffer'
          [value.buffer]
        );
        offset += size;
      }
    });
  }
  fileReadResponse(name, offset, size) {
    return __async(this, null, function* () {
      var _a;
      try {
        const blob = this.fileBlobs.get(name);
        if (!blob) {
          throw new Error(`blob not found for name="${name}"`);
        }
        const chunk = blob.slice(offset, offset + size);
        const buffer = yield chunk.arrayBuffer();
        this.worker.postMessage(
          { verb: "fs.read_res", args: [buffer] },
          { transfer: [buffer] }
        );
      } catch (err) {
        this.logger.error("fileReadResponse failed, terminating worker:", err);
        (_a = this.worker) == null ? void 0 : _a.terminate();
        this.worker = void 0;
        this.abort(`File read failed: ${err}`, err.stack || "");
      }
    });
  }
  /**
   * Parse JSON result returned by cpp code.
   * Throw new Error if "__exception" is present in the response
   *
   * TODO: get rid of this function once everything is migrated to Glue
   */
  parseResult(result) {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult["error"]) {
      throw new WllamaRuntimeError("Unknown error, please see console.log", "");
    }
    return parsedResult;
  }
  /**
   * Push a new task to taskQueue
   */
  pushTask(param, buffers) {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ resolve, reject, param, buffers });
      this.runTaskLoop();
    });
  }
  /**
   * Main loop for processing tasks
   */
  runTaskLoop() {
    return __async(this, null, function* () {
      var _a;
      if (this.busy) {
        return;
      }
      this.busy = true;
      while (true) {
        const task = this.taskQueue.shift();
        if (!task) break;
        this.resultQueue.push(task);
        this.worker.postMessage(
          task.param,
          isSafariMobile() ? void 0 : {
            transfer: (_a = task.buffers) != null ? _a : []
          }
        );
      }
      this.busy = false;
    });
  }
  /**
   * Handle messages from worker
   */
  onRecvMsg(e) {
    if (!e.data) return;
    const { verb, args } = e.data;
    const isCompatBuild = this.resources.compat;
    if (verb && verb.startsWith("console.")) {
      if (this.suppressNativeLog) {
        return;
      }
      if (verb.endsWith("debug")) this.logger.debug(...args);
      if (verb.endsWith("log")) this.logger.log(...args);
      if (verb.endsWith("warn")) this.logger.warn(...args);
      if (verb.endsWith("error")) this.logger.error(...args);
      return;
    } else if (verb === "signal.abort") {
      const [signalType, message, rawStack, originalErr] = args;
      if (originalErr) {
        this.logger.error(originalErr);
      }
      (() => __async(this, null, function* () {
        let stack = "";
        let newMsg = message.replace(
          "Build with -sASSERTIONS for more info.",
          ""
        );
        if (signalType === "abort") {
          newMsg = `(ABORT) ${newMsg}`;
          stack = rawStack.replace(/\|/g, "\n");
        } else if (signalType === "exception") {
          stack = rawStack;
        }
        const decoded = yield Debug.decodeStackTrace(stack, isCompatBuild);
        this.logger.error(`Stack trace (${signalType}):
` + decoded);
        this.abort(newMsg, decoded);
      }))();
      return;
    }
    if (verb === FILE_READ_REQ_EVENT) {
      const [name, offset, size] = args;
      this.fileReadResponse(name, offset, size).catch(() => {
      });
      return;
    }
    const { callbackId, result, err } = e.data;
    if (callbackId) {
      const idx = this.resultQueue.findIndex(
        (t) => t.param.callbackId === callbackId
      );
      if (idx !== -1) {
        const waitingTask = this.resultQueue.splice(idx, 1)[0];
        if (err) waitingTask.reject(err);
        else waitingTask.resolve(result);
      } else {
        this.logger.error(
          `Cannot find waiting task with callbackId = ${callbackId}`
        );
      }
    }
  }
  abort(text, stack) {
    const error = new WllamaRuntimeError(
      text.length == 0 ? "(unknown error)" : text,
      stack
    );
    while (this.resultQueue.length > 0) {
      const waitingTask = this.resultQueue.pop();
      if (!waitingTask) break;
      waitingTask.reject(error);
    }
    while (this.taskQueue.length > 0) {
      const pendingTask = this.taskQueue.pop();
      if (!pendingTask) break;
      pendingTask.reject(error);
    }
  }
};

// src/huggingface.ts
var HF_BASE = "https://huggingface.co";
var DEFAULT_QUANTS = ["Q4_K_M", "Q8_0"];
function fetchRepoFiles(repo, token) {
  return __async(this, null, function* () {
    var _a;
    const url = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
    const headers = { Accept: "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const res = yield fetch(url, { headers });
    if (!res.ok) {
      let msg = res.statusText;
      try {
        msg = (_a = (yield res.json()).error) != null ? _a : msg;
      } catch (e) {
      }
      throw new Error(`HF API error (${res.status}): ${msg}`);
    }
    return res.json();
  });
}
function firstShardPath(files, path) {
  const m = path.match(/^(.+)-(\d{5})-of-(\d{5})\.gguf$/i);
  if (!m) return path;
  const first = `${m[1]}-00001-of-${m[3]}.gguf`;
  return files.some((f) => f.path === first) ? first : path;
}
function selectFile(files, quant, mmprojOnly) {
  const candidates = files.filter((f) => {
    if (f.type !== "file" || !f.path.toLowerCase().endsWith(".gguf"))
      return false;
    const ismmproj = f.path.toLowerCase().includes("mmproj");
    return mmprojOnly ? ismmproj : !ismmproj;
  });
  if (candidates.length === 0) return null;
  if (quant) {
    const upper = quant.toUpperCase();
    const match = candidates.find((f) => f.path.toUpperCase().includes(upper));
    if (match) return firstShardPath(candidates, match.path);
    return null;
  }
  for (const q of DEFAULT_QUANTS) {
    const match = candidates.find((f) => f.path.toUpperCase().includes(q));
    if (match) return firstShardPath(candidates, match.path);
  }
  return firstShardPath(candidates, candidates[0].path);
}
function getHFModelSource(config) {
  return __async(this, null, function* () {
    const { repo, file, quant, mmprojFile, mmprojQuant, hfToken } = config;
    const files = yield fetchRepoFiles(repo, hfToken);
    const modelPath = file != null ? file : selectFile(files, quant, false);
    if (!modelPath) {
      throw new Error(`No GGUF file found in repo "${repo}"`);
    }
    const source = {
      url: `${HF_BASE}/${repo}/resolve/main/${modelPath}`
    };
    if (mmprojFile || mmprojQuant !== void 0) {
      const mmpath = mmprojFile != null ? mmprojFile : selectFile(files, mmprojQuant, true);
      if (mmpath) {
        source.mmprojUrl = `${HF_BASE}/${repo}/resolve/main/${mmpath}`;
      }
    }
    if (hfToken) {
      const params = new URLSearchParams({ token: hfToken });
      source.url += `?${params}`;
      if (source.mmprojUrl) {
        source.mmprojUrl += `?${params}`;
      }
    }
    return source;
  });
}
function getHFFileSHA256(url, headers) {
  return __async(this, null, function* () {
    if (!url.includes("/resolve/")) return void 0;
    const rawUrl = url.replace("/resolve/", "/raw/");
    try {
      const text = yield fetch(rawUrl, { headers }).then((r) => r.text());
      const match = text.match(/^oid sha256:([0-9a-f]{64})$/m);
      return match ? match[1] : void 0;
    } catch (e) {
      return void 0;
    }
  });
}

// src/storage/opfs.ts
var OPFSBackend = class {
  isSupported() {
    var _a;
    return typeof navigator !== "undefined" && "storage" in navigator && !!((_a = navigator.storage) == null ? void 0 : _a.getDirectory);
  }
  read(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        return yield fileHandle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  write(key, stream) {
    return __async(this, null, function* () {
      const writable = yield openWritable(key);
      yield writable.truncate(0);
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        const file = yield fileHandle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      const cacheDir = yield getCacheDir();
      const result = [];
      try {
        for (var iter = __forAwait(cacheDir.entries()), more, temp, error; more = !(temp = yield iter.next()).done; more = false) {
          const [name, handle] = temp.value;
          if (handle.kind === "file") {
            const file = yield handle.getFile();
            result.push({ key: name, size: file.size });
          }
        }
      } catch (temp) {
        error = [temp];
      } finally {
        try {
          more && (temp = iter.return) && (yield temp.call(iter));
        } finally {
          if (error)
            throw error[0];
        }
      }
      return result;
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        yield cacheDir.removeEntry(key);
      } catch (e) {
        if ((e == null ? void 0 : e.name) !== "NotFoundError") throw e;
      }
    });
  }
};
function getCacheDir() {
  return __async(this, null, function* () {
    const opfsRoot = yield navigator.storage.getDirectory();
    return opfsRoot.getDirectoryHandle("cache", { create: true });
  });
}
function openWritable(fileName) {
  return __async(this, null, function* () {
    const worker = createWorker(OPFS_UTILS_WORKER_CODE);
    let pResolve;
    let pReject;
    worker.onmessage = (e) => {
      if (e.data.ok) pResolve(null);
      else if (e.data.err) pReject(e.data.err);
    };
    worker.onerror = (e) => {
      var _a;
      return pReject == null ? void 0 : pReject((_a = e.message) != null ? _a : e);
    };
    const workerExec = (data) => new Promise((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      worker.postMessage(
        data,
        isSafariMobile() ? void 0 : { transfer: "buf" in data && data.buf ? [data.buf.buffer] : [] }
      );
    });
    yield workerExec({ action: "open", filename: fileName });
    return {
      truncate: () => __async(this, null, function* () {
      }),
      write: (value) => workerExec({ action: "write", buf: value }),
      close: () => __async(this, null, function* () {
        yield workerExec({ action: "close" });
        worker.terminate();
      })
    };
  });
}

// src/storage/cos.ts
function makeHash(key) {
  return { algorithm: "SHA-256", value: key };
}
var COSInternalBackend = class {
  isSupported() {
    return typeof navigator !== "undefined" && "crossOriginStorage" in navigator;
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  read(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        return handle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  write(key, stream) {
    return __async(this, null, function* () {
      const handle = yield navigator.crossOriginStorage.requestFileHandle(
        makeHash(key),
        { create: true }
      );
      const writable = yield handle.createWritable();
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        const file = yield handle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
  delete(_key) {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
};
var COSBackend = class {
  constructor() {
    __publicField(this, "cos", new COSInternalBackend());
    __publicField(this, "priv", new OPFSBackend());
  }
  isSupported() {
    return this.priv.isSupported();
  }
  read(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const blob = yield this.cos.read(hint.sha256);
        if (blob) return blob;
      }
      return this.priv.read(key);
    });
  }
  write(key, stream, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        yield this.cos.write(hint.sha256, stream);
      } else {
        yield this.priv.write(key, stream);
      }
    });
  }
  getSize(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const size = yield this.cos.getSize(hint.sha256);
        if (size !== -1) return size;
      }
      return this.priv.getSize(key);
    });
  }
  list() {
    return __async(this, null, function* () {
      return this.priv.list();
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      return this.priv.delete(key);
    });
  }
};

// src/cache-manager.ts
var PREFIX_METADATA = "__metadata__";
var POLYFILL_ETAG = "polyfill_for_older_version";
function hintFromMetadata(metadata) {
  if (!metadata) return void 0;
  if (metadata.sha256) return { sha256: metadata.sha256 };
  return void 0;
}
var CacheManager = class {
  /**
   * @param backends Array of storage backends to use, in order of preference ; if first is available, use it, otherwise try the next one.
   */
  constructor(backends = [new COSBackend()]) {
    __publicField(this, "sb");
    for (const backend of backends) {
      if (backend.isSupported()) {
        this.sb = backend;
        return;
      }
    }
    throw new Error("No supported storage backend found");
  }
  /**
   * Convert a given URL into a storage key.
   *
   * Format: `${hashSHA1(fullURL)}_${fileName}`
   */
  getNameFromURL(url) {
    return __async(this, null, function* () {
      return urlToFileName(url, "");
    });
  }
  /**
   * @deprecated Use `download()` instead
   *
   * Write a new file to cache. This will overwrite existing file.
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   */
  write(name, stream, metadata) {
    return __async(this, null, function* () {
      yield this.sb.write(name, stream);
      yield this.writeMetadata(name, metadata);
    });
  }
  download(_0) {
    return __async(this, arguments, function* (url, options = {}) {
      var _a, _b, _c, _d;
      const fileKey = yield urlToFileName(url, "");
      const sha256 = yield getHFFileSHA256(url, (_a = options.headers) != null ? _a : {});
      const hint = sha256 ? { sha256 } : void 0;
      const cachedSize = yield this.sb.getSize(fileKey, hint);
      if (cachedSize !== -1) {
        const metadata2 = yield this.readMetadata(fileKey);
        if ((metadata2 == null ? void 0 : metadata2.originalURL) === url && metadata2.originalSize === cachedSize) {
          return;
        }
        const head = yield fetch(url, __spreadValues({
          method: "HEAD"
        }, options.headers ? { headers: options.headers } : {}));
        const originalSize = parseInt(
          (_b = head.headers.get("content-length")) != null ? _b : "0",
          10
        );
        const etag2 = (head.headers.get("etag") || "").replace(
          /[^A-Za-z0-9]/g,
          ""
        );
        if (originalSize > 0 && originalSize === cachedSize) {
          yield this.writeMetadata(fileKey, __spreadValues({
            originalURL: url,
            originalSize,
            etag: etag2,
            sha256
          }, (_c = options.metadataAdditional) != null ? _c : {}));
          return;
        }
        yield this.sb.delete(fileKey);
        yield this.sb.delete(`${PREFIX_METADATA}${fileKey}`);
      }
      const response = yield fetch(url, __spreadValues(__spreadValues({}, options.headers ? { headers: options.headers } : {}), options.signal ? { signal: options.signal } : {}));
      if (!response.ok || !response.body) {
        throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
      }
      const contentLength = response.headers.get("content-length");
      const etag = (response.headers.get("etag") || "").replace(
        /[^A-Za-z0-9]/g,
        ""
      );
      const total = parseInt(contentLength != null ? contentLength : "0", 10);
      const progressCallback = options.progressCallback;
      let loaded = 0;
      let lastProgressAt = 0;
      const progressStream = new TransformStream({
        transform(chunk, controller) {
          loaded += chunk.byteLength;
          if (progressCallback) {
            const now = Date.now();
            if (now - lastProgressAt > 100) {
              lastProgressAt = now;
              progressCallback({ loaded, total });
            }
          }
          controller.enqueue(chunk);
        },
        flush() {
          progressCallback == null ? void 0 : progressCallback({ loaded, total: total || loaded });
        }
      });
      const metadata = __spreadValues({
        originalURL: url,
        originalSize: total,
        etag
      }, (_d = options.metadataAdditional) != null ? _d : {});
      if (sha256) {
        metadata.sha256 = sha256;
      }
      yield this.sb.write(
        fileKey,
        response.body.pipeThrough(progressStream),
        hint
      );
      yield this.writeMetadata(fileKey, metadata);
    });
  }
  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  open(nameOrURL) {
    return __async(this, null, function* () {
      const hint1 = hintFromMetadata(yield this.getMetadata(nameOrURL));
      const direct = yield this.sb.read(nameOrURL, hint1);
      if (direct) return direct;
      const key = yield urlToFileName(nameOrURL, "");
      const hint2 = hintFromMetadata(yield this.getMetadata(key));
      return this.sb.read(key, hint2);
    });
  }
  /**
   * Get the size of a file in stored cache
   *
   * NOTE: in case the download is stopped mid-way (i.e. user close browser tab), the file maybe corrupted, size maybe different from `metadata.originalSize`
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   * @returns number of bytes, or -1 if file does not exist
   */
  getSize(name) {
    return __async(this, null, function* () {
      const hint = hintFromMetadata(yield this.getMetadata(name));
      return this.sb.getSize(name, hint);
    });
  }
  /**
   * Get metadata of a cached file
   */
  getMetadata(name) {
    return __async(this, null, function* () {
      const metadata = yield this.readMetadata(name);
      if (metadata) return metadata;
      const cachedSize = yield this.sb.getSize(name);
      return cachedSize > 0 ? (
        // files created by older version of wllama don't have metadata; polyfill it
        {
          etag: POLYFILL_ETAG,
          originalSize: cachedSize,
          originalURL: ""
        }
      ) : (
        // cached file not found
        null
      );
    });
  }
  /**
   * Same as `getMetadata()`, but without polyfill. Returns null if the file has no metadata.
   */
  readMetadata(name) {
    return __async(this, null, function* () {
      const blob = yield this.sb.read(`${PREFIX_METADATA}${name}`);
      if (!blob) return null;
      try {
        return yield new Response(blob).json();
      } catch (e) {
        return null;
      }
    });
  }
  /**
   * List all files currently in cache
   */
  list() {
    return __async(this, null, function* () {
      const all = yield this.sb.list();
      const metadataMap = {};
      for (const { key } of all) {
        if (key.startsWith(PREFIX_METADATA)) {
          const blob = yield this.sb.read(key);
          if (blob) {
            const meta = yield new Response(blob).json().catch(() => null);
            metadataMap[key.slice(PREFIX_METADATA.length)] = meta;
          }
        }
      }
      const result = [];
      for (const { key, size } of all) {
        if (!key.startsWith(PREFIX_METADATA)) {
          result.push({
            name: key,
            size,
            metadata: metadataMap[key] || {
              originalSize: size,
              originalURL: "",
              etag: ""
            }
          });
        }
      }
      return result;
    });
  }
  /**
   * Clear all files currently in cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.deleteMany(() => true);
    });
  }
  /**
   * Delete a single file in cache
   *
   * @param nameOrURL Can be either an URL or a name returned by `getNameFromURL()` or `list()`
   */
  delete(nameOrURL) {
    return __async(this, null, function* () {
      const name2 = yield this.getNameFromURL(nameOrURL);
      yield this.deleteMany(
        (entry) => entry.name === nameOrURL || entry.name === name2
      );
    });
  }
  /**
   * Delete multiple files in cache.
   *
   * @param predicate A predicate like `array.filter(item => boolean)`
   */
  deleteMany(predicate) {
    return __async(this, null, function* () {
      const list = yield this.list();
      for (const item of list) {
        if (predicate(item)) {
          yield this.sb.delete(item.name);
          yield this.sb.delete(`${PREFIX_METADATA}${item.name}`);
        }
      }
    });
  }
  /**
   * Write the metadata of the file to disk.
   */
  writeMetadata(name, metadata) {
    return __async(this, null, function* () {
      const blob = new Blob([JSON.stringify(metadata)], { type: "text/plain" });
      yield this.sb.write(`${PREFIX_METADATA}${name}`, blob.stream());
    });
  }
};
var cache_manager_default = CacheManager;
function urlToFileName(url, prefix) {
  return __async(this, null, function* () {
    const hashBuffer = yield crypto.subtle.digest(
      "SHA-1",
      new TextEncoder().encode(url)
    );
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
    return `${prefix}${hashHex}_${url.split("/").pop()}`;
  });
}

// src/model-manager.ts
var DEFAULT_PARALLEL_DOWNLOADS = 3;
var ModelValidationStatus = /* @__PURE__ */ ((ModelValidationStatus2) => {
  ModelValidationStatus2["VALID"] = "valid";
  ModelValidationStatus2["INVALID"] = "invalid";
  ModelValidationStatus2["DELETED"] = "deleted";
  return ModelValidationStatus2;
})(ModelValidationStatus || {});
var Model = class {
  constructor(modelManager, url, mmprojUrl, savedFiles) {
    __publicField(this, "modelManager");
    /**
     * URL to the GGUF file (in case it contains multiple shards, the URL should point to the first shard)
     *
     * This URL will be used to identify the model in the cache. There can't be 2 models with the same URL.
     */
    __publicField(this, "url");
    /**
     * URL to mmproj file, if exists
     */
    __publicField(this, "mmprojUrl");
    /**
     * Size in bytes (total size of all shards).
     *
     * A value of -1 means the model is deleted from the cache. You must call `ModelManager.downloadModel` to re-download the model.
     */
    __publicField(this, "size");
    /**
     * List of all shards in the cache, sorted by original URL (ascending order)
     */
    __publicField(this, "files");
    this.modelManager = modelManager;
    this.url = url;
    this.mmprojUrl = mmprojUrl;
    if (savedFiles) {
      this.files = this.getAllFiles(savedFiles);
      this.size = sumArr(this.files.map((f) => f.metadata.originalSize));
    } else {
      this.files = [];
      this.size = 0;
    }
  }
  /**
   * Open and get a list of all shards as Blobs
   */
  open() {
    return __async(this, null, function* () {
      if (this.size === -1) {
        throw new WllamaError(
          `Model is deleted from the cache; Call ModelManager.downloadModel to re-download the model`,
          "load_error"
        );
      }
      const blobs = [];
      for (const file of this.files) {
        const blob = yield this.modelManager.cacheManager.open(file.name);
        if (!blob) {
          throw new Error(
            `Failed to open file ${file.name}; Hint: the model may be invalid, please refresh it`
          );
        }
        blobs.push(blob);
      }
      return blobs;
    });
  }
  /**
   * Validate the model files.
   *
   * If the model is invalid, the model manager will not be able to use it. You must call `refresh` to re-download the model.
   *
   * Cases that model is invalid:
   * - The model is deleted from the cache
   * - The model files are missing (or the download is interrupted)
   */
  validate() {
    let nbShards = ModelManager.parseModelUrl(this.url).length;
    if (this.mmprojUrl) {
      nbShards += 1;
    }
    if (this.size === -1) {
      return "deleted" /* DELETED */;
    }
    if (this.size < 16 || this.files.length !== nbShards) {
      return "invalid" /* INVALID */;
    }
    for (const file of this.files) {
      if (!file.metadata || file.metadata.originalSize !== file.size) {
        return "invalid" /* INVALID */;
      }
    }
    return "valid" /* VALID */;
  }
  /**
   * In case the model is invalid, call this function to re-download the model
   */
  refresh() {
    return __async(this, arguments, function* (options = {}) {
      var _a;
      const urls = ModelManager.parseModelUrl(this.url);
      if (this.mmprojUrl) {
        urls.push(this.mmprojUrl);
      }
      const works = urls.map((url, index) => ({
        url,
        index
      }));
      this.modelManager.logger.debug("Downloading model files:", urls);
      const nParallel = (_a = this.modelManager.params.parallelDownloads) != null ? _a : DEFAULT_PARALLEL_DOWNLOADS;
      const totalSize = yield this.getTotalDownloadSize(urls);
      const loadedSize = [];
      const worker = () => __async(this, null, function* () {
        while (works.length > 0) {
          const w = works.shift();
          if (!w) break;
          yield this.modelManager.cacheManager.download(w.url, __spreadProps(__spreadValues({}, options), {
            metadataAdditional: {
              originalURL: w.url,
              mmprojURL: this.mmprojUrl
            },
            progressCallback: ({ loaded }) => {
              var _a2;
              loadedSize[w.index] = loaded;
              (_a2 = options.progressCallback) == null ? void 0 : _a2.call(options, {
                loaded: sumArr(loadedSize),
                total: totalSize
              });
            }
          }));
        }
      });
      const promises = [];
      for (let i = 0; i < nParallel; i++) {
        promises.push(worker());
        loadedSize.push(0);
      }
      yield Promise.all(promises);
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      this.size = this.files.reduce((acc, f) => acc + f.metadata.originalSize, 0);
    });
  }
  /**
   * Remove the model from the cache
   */
  remove() {
    return __async(this, null, function* () {
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      yield this.modelManager.cacheManager.deleteMany(
        (f) => !!this.files.find((file) => file.name === f.name)
      );
      this.size = -1;
    });
  }
  getAllFiles(savedFiles) {
    const allUrls = new Set(ModelManager.parseModelUrl(this.url));
    if (this.mmprojUrl) {
      allUrls.add(this.mmprojUrl);
    }
    const allFiles = [];
    for (const url of allUrls) {
      const file = savedFiles.find((f) => f.metadata.originalURL === url);
      if (!file) {
        throw new Error(`Model file not found: ${url}`);
      }
      allFiles.push(file);
    }
    allFiles.sort(
      (a, b) => a.metadata.originalURL.localeCompare(b.metadata.originalURL)
    );
    return allFiles;
  }
  getTotalDownloadSize(urls) {
    return __async(this, null, function* () {
      const responses = yield Promise.all(
        urls.map((url) => fetch(url, { method: "HEAD" }))
      );
      const sizes = responses.map(
        (res) => Number(res.headers.get("content-length") || "0")
      );
      return sumArr(sizes);
    });
  }
};
var ModelManager = class _ModelManager {
  constructor(params = {}) {
    // The CacheManager singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "params");
    __publicField(this, "logger");
    this.cacheManager = params.cacheManager || new cache_manager_default();
    this.params = params;
    this.logger = params.logger || console;
  }
  /**
   * Parses a model URL and returns an array of URLs based on the following patterns:
   * - If the input URL is an array, it returns the array itself.
   * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
   * - Otherwise, it returns an array containing the input URL as a single element array.
   * @param modelUrl URL or list of URLs
   */
  static parseModelUrl(modelUrl) {
    var _a;
    if (Array.isArray(modelUrl)) {
      return modelUrl;
    }
    const urlPartsRegex = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
    const queryMatch = modelUrl.match(/\.gguf(\?.*)?$/);
    const queryParams = (_a = queryMatch == null ? void 0 : queryMatch[1]) != null ? _a : "";
    const matches = modelUrl.match(urlPartsRegex);
    if (!matches) {
      return [modelUrl];
    }
    const baseURL = modelUrl.replace(urlPartsRegex, "");
    const total = matches[2];
    const paddedShardIds = Array.from(
      { length: Number(total) },
      (_, index) => (index + 1).toString().padStart(5, "0")
    );
    return paddedShardIds.map(
      (current) => `${baseURL}-${current}-of-${total}.gguf${queryParams}`
    );
  }
  /**
   * Get all models in the cache
   */
  getModels() {
    return __async(this, arguments, function* (opts = {}) {
      const cachedFiles = yield this.cacheManager.list();
      let models = [];
      for (const file of cachedFiles) {
        if (!file.metadata.originalURL) continue;
        const shards = _ModelManager.parseModelUrl(file.metadata.originalURL);
        const mmprojUrl = file.metadata.mmprojURL;
        const isFirstShard = shards.length === 1 || shards[0] === file.metadata.originalURL;
        if (isFirstShard) {
          models.push(
            new Model(this, file.metadata.originalURL, mmprojUrl, cachedFiles)
          );
        }
      }
      if (!opts.includeInvalid) {
        models = models.filter(
          (m) => m.validate() === "valid" /* VALID */
        );
      }
      return models;
    });
  }
  /**
   * Download a model from the given URL.
   *
   * The URL must end with `.gguf`
   */
  downloadModel(_0) {
    return __async(this, arguments, function* (sourceOrURL, options = {}) {
      const source = isString(sourceOrURL) ? { url: sourceOrURL } : sourceOrURL;
      if (!isValidGgufFile(source.url)) {
        throw new WllamaError(
          `Invalid model URL: ${source.url}; URL must ends with ".gguf"`,
          "download_error"
        );
      }
      const model = new Model(this, source.url, source.mmprojUrl);
      const validity = model.validate();
      if (validity !== "valid" /* VALID */) {
        yield model.refresh(options);
      }
      return model;
    });
  }
  /**
   * Get a model from the cache or download it if it's not available.
   */
  getModelOrDownload(_0) {
    return __async(this, arguments, function* (source, options = {}) {
      var _a;
      const models = yield this.getModels();
      const model = models.find((m) => m.url === source.url);
      if (model) {
        (_a = options.progressCallback) == null ? void 0 : _a.call(options, { loaded: model.size, total: model.size });
        return model;
      }
      return this.downloadModel(source, options);
    });
  }
  /**
   * Remove all models from the cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.cacheManager.clear();
    });
  }
};

// src/types/types.ts
var LogLevel = /* @__PURE__ */ ((LogLevel2) => {
  LogLevel2[LogLevel2["DEBUG"] = 1] = "DEBUG";
  LogLevel2[LogLevel2["INFO"] = 2] = "INFO";
  LogLevel2[LogLevel2["WARN"] = 3] = "WARN";
  LogLevel2[LogLevel2["ERROR"] = 4] = "ERROR";
  return LogLevel2;
})(LogLevel || {});

// src/wasm-from-cdn.ts
var WasmCompatFromCDN = {
  worker: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.js",
  wasm: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.wasm"
};

// src/wllama.ts
var LoggerWithoutDebug = __spreadProps(__spreadValues({}, console), {
  debug: () => {
  }
});
var WllamaError = class extends Error {
  constructor(message, type = "unknown_error") {
    super(message);
    __publicField(this, "type");
    this.type = type;
  }
};
var WllamaAbortError = class extends Error {
  constructor() {
    super("Operation aborted");
    __publicField(this, "name", "AbortError");
  }
};
var WllamaRuntimeError = class extends Error {
  constructor(message, stack) {
    super(message);
    __publicField(this, "name", "RuntimeError");
    __publicField(this, "stack");
    this.stack = stack;
  }
};
var Wllama = class {
  constructor(pathConfig, wllamaConfig = {}) {
    // The CacheManager and ModelManager are singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "modelManager");
    /** Feature detection: `loraBlobs` means `lora_adapters` accepts GGUF blobs, `nativeStates` means `native_states` loads natlang native state spans, `natlangScore` means `createNatlangScore` is available, `slotState` means `saveSlot` / `restoreSlot` / `eraseSlot` / `applyTemplate` are available and completions accept `id_slot`, `nativeStateStore` means `putNativeState` / `getNativeState` are available. */
    __publicField(this, "capabilities", {
      loraBlobs: true,
      nativeStates: true,
      natlangScore: true,
      slotState: true,
      nativeStateStore: true
    });
    __publicField(this, "slotFileId", 0);
    __publicField(this, "compat", null);
    __publicField(this, "proxy", null);
    __publicField(this, "config");
    __publicField(this, "pathConfig");
    __publicField(this, "useMultiThread", false);
    __publicField(this, "nbThreads", 1);
    __publicField(this, "useEmbeddings", false);
    __publicField(this, "useRerank", false);
    // available when loaded
    __publicField(this, "loadedContextInfo", null);
    __publicField(this, "seed");
    __publicField(this, "bosToken", -1);
    __publicField(this, "eosToken", -1);
    __publicField(this, "eotToken", -1);
    __publicField(this, "eogTokens", /* @__PURE__ */ new Set());
    __publicField(this, "addBosToken", false);
    __publicField(this, "addEosToken", false);
    __publicField(this, "mediaMarker");
    __publicField(this, "chatTemplate");
    __publicField(this, "metadata");
    __publicField(this, "hasEncoder", false);
    __publicField(this, "decoderStartToken", -1);
    // the defaults are also the server's own (default_template_kwargs_keys/vals at load, as JSON texts), which is what
    // scoring renders with; the overlay keeps them on chat requests, whose own kwargs win
    __publicField(this, "chatTemplateKwargs", {});
    var _a, _b, _c;
    checkEnvironmentCompatible();
    if (!pathConfig) throw new WllamaError("AssetsPathConfig is required");
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
    this.cacheManager = (_a = wllamaConfig.cacheManager) != null ? _a : new cache_manager_default();
    this.modelManager = (_c = wllamaConfig.modelManager) != null ? _c : new ModelManager({
      cacheManager: this.cacheManager,
      logger: (_b = wllamaConfig.logger) != null ? _b : console,
      parallelDownloads: wllamaConfig.parallelDownloads,
      allowOffline: wllamaConfig.allowOffline
    });
    this.setCompat("default");
  }
  logger() {
    var _a;
    return (_a = this.config.logger) != null ? _a : console;
  }
  checkModelLoaded() {
    if (!this.isModelLoaded()) {
      throw new WllamaError(
        "loadModel() is not yet called",
        "model_not_loaded"
      );
    }
  }
  /**
   * Get the libllama version string, e.g. "b6327-4d74393".
   *
   * @returns version string embedded at build time.
   */
  static getLibllamaVersion() {
    return LIBLLAMA_VERSION;
  }
  /**
   * Set compatibility options for Wllama.
   * @param compat Set to null to disable compatibility, or 'default' to use the default compat resources from CDN.
   * @param mode 'safari' by default; If set to 'firefox_safari', the compat mode will **also** be enabled on Firefox, which will significantly degrade the performance but allow using WebGPU on Firefox.
   */
  setCompat(compat, mode = "safari") {
    if (mode === "safari") {
      if (isFirefox()) {
        this.compat = null;
        return;
      }
    }
    this.compat = compat === "default" ? WasmCompatFromCDN : compat;
  }
  /**
   * Check if the model is loaded via `loadModel()`
   */
  isModelLoaded() {
    return !!this.proxy && !!this.metadata;
  }
  /**
   * Get token ID associated to BOS (begin of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getBOS() {
    return this.bosToken;
  }
  /**
   * Get token ID associated to EOS (end of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOS() {
    return this.eosToken;
  }
  /**
   * Get token ID associated to EOT (end of turn) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOT() {
    return this.eotToken;
  }
  /**
   * Check if a given token is end-of-generation token (e.g. EOS, EOT, etc.)
   *
   * @param token the token ID to be checked
   * @returns true if the token is EOS, EOT, or any other end-of-generation tokens
   */
  isTokenEOG(token) {
    return token === this.eosToken || token === this.eotToken || this.eogTokens.has(token);
  }
  /**
   * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getDecoderStartToken() {
    return this.decoderStartToken;
  }
  /**
   * Get model hyper-parameters and metadata
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns ModelMetadata
   */
  getModelMetadata() {
    this.checkModelLoaded();
    return this.metadata;
  }
  /**
   * Check if we're currently using multi-thread build.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isMultithread() {
    this.checkModelLoaded();
    return this.useMultiThread;
  }
  /**
   * Get number of threads used in the current context.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns number of threads
   */
  getNumThreads() {
    this.checkModelLoaded();
    return this.useMultiThread ? this.nbThreads : 1;
  }
  /**
   * Check if the current model uses encoder-decoder architecture
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isEncoderDecoderArchitecture() {
    this.checkModelLoaded();
    return this.hasEncoder;
  }
  /**
   * Must we add BOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if BOS token must be added to the sequence
   */
  mustAddBosToken() {
    this.checkModelLoaded();
    return this.addBosToken;
  }
  /**
   * Must we add EOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if EOS token must be added to the sequence
   */
  mustAddEosToken() {
    this.checkModelLoaded();
    return this.addEosToken;
  }
  /**
   * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns the jinja template. null if there is no template in gguf
   */
  getChatTemplate() {
    var _a;
    this.checkModelLoaded();
    return (_a = this.chatTemplate) != null ? _a : null;
  }
  /**
   * Check if WebGPU is supported by the current environment.
   * @returns true if WebGPU is supported
   */
  isSupportWebGPU() {
    return isSupportWebGPU();
  }
  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelSourceOrURL
   * @param params
   */
  loadModelFromUrl(_0) {
    return __async(this, arguments, function* (modelSourceOrURL, params = {}) {
      var _a;
      const source = isString(modelSourceOrURL) ? { url: modelSourceOrURL } : modelSourceOrURL;
      const useCache = (_a = params.useCache) != null ? _a : true;
      const model = useCache ? yield this.modelManager.getModelOrDownload(source, params) : yield this.modelManager.downloadModel(source, params);
      const blobs = yield model.open();
      return yield this.loadModel(blobs, params);
    });
  }
  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param hfOptions
   * @param params
   */
  loadModelFromHF(_0) {
    return __async(this, arguments, function* (hfOptions, params = {}) {
      const source = yield getHFModelSource(hfOptions);
      return yield this.loadModelFromUrl(source, params);
    });
  }
  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param params LoadModelParams
   */
  loadModel(_0) {
    return __async(this, arguments, function* (ggufBlobsOrModel, params = {}) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k;
      const blobs = ggufBlobsOrModel instanceof Model ? yield ggufBlobsOrModel.open() : [...ggufBlobsOrModel];
      if (blobs.some((b) => b.size === 0)) {
        throw new WllamaError(
          "Input model (or splits) must be non-empty Blob or File",
          "load_error"
        );
      }
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (this.proxy) {
        throw new WllamaError("Module is already initialized", "load_error");
      }
      const supportMultiThread = yield isSupportMultiThread();
      const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
      const nbThreads = (_a = params.n_threads) != null ? _a : hwConccurency;
      this.nbThreads = nbThreads;
      this.useMultiThread = supportMultiThread && nbThreads > 1;
      const workerResources = this.getWorkerResources();
      if (params.n_gpu_layers === 0) {
        workerResources.noWebGPU = true;
      }
      this.proxy = new ProxyToWorker(
        workerResources,
        this.useMultiThread ? nbThreads : 0,
        // 0 means disable pthread
        (_b = this.config.suppressNativeLog) != null ? _b : false,
        this.logger()
      );
      let logLevel = (_c = params.log_level) != null ? _c : 2 /* INFO */;
      if (this.config.suppressNativeLog) {
        logLevel = 9999;
      }
      const modelFiles = yield prepareBlobs(blobs);
      const loraPaths = [];
      const sideFiles = [];
      for (const [i, adapter] of ((_d = params.lora_adapters) != null ? _d : []).entries()) {
        if (adapter.blob) {
          if (adapter.blob.size === 0) {
            throw new WllamaError(
              `LoRA adapter ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `lora-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: adapter.blob });
          loraPaths.push(`models/${name}`);
        } else if (adapter.path) {
          loraPaths.push(adapter.path);
        } else {
          throw new WllamaError(
            `LoRA adapter ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      const statePaths = [];
      for (const [i, state] of ((_e = params.native_states) != null ? _e : []).entries()) {
        if (state.blob) {
          if (state.blob.size === 0) {
            throw new WllamaError(
              `Native state ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `native-state-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: state.blob });
          statePaths.push(`models/${name}`);
        } else if (state.path) {
          statePaths.push(state.path);
        } else {
          throw new WllamaError(
            `Native state ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      yield this.proxy.moduleInit([...modelFiles.all, ...sideFiles]);
      this.logger().debug("Calling wllamaStart...");
      const startResult = yield this.proxy.wllamaStart();
      if (!startResult.success) {
        throw new WllamaError(
          `Error while calling start function, result = ${startResult}`
        );
      }
      this.logger().debug("Loading model...");
      const loadResult = yield this.proxy.wllamaAction("load", {
        _name: "load_req",
        log_level: logLevel,
        // if async read is not supported, use mmap; refer to README-dev.md for more details
        use_mmap: !canUseAsyncFileRead(workerResources.compat),
        use_mlock: false,
        n_gpu_layers: (_f = params.n_gpu_layers) != null ? _f : 99999,
        n_ctx: (_g = params.n_ctx) != null ? _g : 1024,
        n_threads: this.useMultiThread ? nbThreads : 1,
        n_ctx_auto: false,
        // not supported for now
        mmproj_path: modelFiles.mmproj ? `/models/${MMPROJ_FILE_NAME}` : void 0,
        model_paths: modelFiles.llm.map((f) => `models/${f.name}`),
        embeddings: params.embeddings,
        offload_kqv: params.offload_kqv,
        n_batch: params.n_batch,
        n_ubatch: params.n_ubatch,
        pooling_type: params.pooling_type,
        rope_scaling_type: params.rope_scaling_type,
        rope_freq_base: params.rope_freq_base,
        rope_freq_scale: params.rope_freq_scale,
        yarn_ext_factor: params.yarn_ext_factor,
        yarn_attn_factor: params.yarn_attn_factor,
        yarn_beta_fast: params.yarn_beta_fast,
        yarn_beta_slow: params.yarn_beta_slow,
        yarn_orig_ctx: params.yarn_orig_ctx,
        cache_type_k: params.cache_type_k,
        cache_type_v: params.cache_type_v,
        // with unified KV, all sequences share one n_ctx cache, so each request can still use the full context
        n_parallel: (_h = params.n_parallel) != null ? _h : 4,
        kv_unified: (_i = params.kv_unified) != null ? _i : true,
        flash_attn: params.flash_attn,
        swa_full: params.swa_full,
        chat_template: params.chat_template,
        jinja: params.jinja,
        reasoning: params.reasoning,
        image_min_tokens: params.image_min_tokens,
        image_max_tokens: params.image_max_tokens,
        warmup: params.warmup,
        no_kv_offload: params.no_kv_offload,
        mmproj_offload: params.mmproj_offload,
        cont_batching: params.cont_batching,
        n_keep: params.n_keep,
        ctx_shift: params.ctx_shift,
        cache_idle_slots: params.cache_idle_slots,
        n_cache_reuse: params.n_cache_reuse,
        lora_paths: params.lora_adapters ? loraPaths : void 0,
        lora_scales: (_j = params.lora_adapters) == null ? void 0 : _j.map((a) => {
          var _a2;
          return (_a2 = a.scale) != null ? _a2 : 1;
        }),
        lora_init_without_apply: params.lora_init_without_apply,
        native_state_paths: params.native_states ? statePaths : void 0,
        spec_draft_model: params.spec_draft_model,
        spec_draft_ngl: params.spec_draft_ngl,
        spec_draft_n_max: params.spec_draft_n_max,
        spec_draft_n_min: params.spec_draft_n_min,
        spec_draft_p_min: params.spec_draft_p_min,
        spec_draft_threads: params.spec_draft_threads,
        spec_draft_threads_batch: params.spec_draft_threads_batch,
        kv_overrides_keys: params.kv_overrides ? Object.keys(params.kv_overrides) : void 0,
        kv_overrides_vals: params.kv_overrides ? Object.values(params.kv_overrides) : void 0,
        // the server's own defaults too, so every request it renders gets them (natlang's /v1/natlang/score items as
        // well as chat), as llama-server's --chat-template-kwargs does; each value travels as its JSON text, which is
        // how the server stores them
        default_template_kwargs_keys: params.default_template_kwargs ? Object.keys(params.default_template_kwargs) : void 0,
        default_template_kwargs_vals: params.default_template_kwargs ? Object.values(params.default_template_kwargs).map((value) => JSON.stringify(value)) : void 0,
        reasoning_budget_tokens: params.reasoning_budget_tokens,
        reasoning_budget_message: params.reasoning_budget_message,
        reasoning_format: params.reasoning_format,
        skip_chat_parsing: params.skip_chat_parsing,
        prefill_assistant: params.prefill_assistant
      });
      const loadedCtxInfo = __spreadProps(__spreadValues({}, loadResult), {
        metadata: {}
      });
      for (let i = 0; i < loadResult.metadata_key.length; i++) {
        loadedCtxInfo.metadata[loadResult.metadata_key[i]] = loadResult.metadata_val[i];
      }
      this.seed = params.seed;
      this.bosToken = loadedCtxInfo.token_bos;
      this.eosToken = loadedCtxInfo.token_eos;
      this.eotToken = loadedCtxInfo.token_eot;
      this.useEmbeddings = !!params.embeddings;
      this.useRerank = params.pooling_type == "rank";
      this.metadata = {
        hparams: {
          nVocab: loadedCtxInfo.n_vocab,
          nCtxTrain: loadedCtxInfo.n_ctx_train,
          nEmbd: loadedCtxInfo.n_embd,
          nLayer: loadedCtxInfo.n_layer
        },
        meta: loadedCtxInfo.metadata
      };
      this.hasEncoder = !!loadedCtxInfo.has_encoder;
      this.decoderStartToken = loadedCtxInfo.token_decoder_start;
      this.addBosToken = loadedCtxInfo.add_bos_token;
      this.addEosToken = loadedCtxInfo.add_eos_token;
      this.chatTemplate = loadedCtxInfo.metadata["tokenizer.chat_template"];
      this.loadedContextInfo = loadedCtxInfo;
      this.eogTokens = new Set(loadedCtxInfo.list_tokens_eog);
      this.mediaMarker = loadedCtxInfo.media_marker;
      this.chatTemplateKwargs = (_k = params.default_template_kwargs) != null ? _k : {};
      this.logger().debug({ loadedCtxInfo });
    });
  }
  getLoadedContextInfo() {
    this.checkModelLoaded();
    if (!this.loadedContextInfo) {
      throw new WllamaError("Loaded context info is not available");
    }
    return __spreadValues({}, this.loadedContextInfo);
  }
  //////////////////////////////////////////////
  // High level API
  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param options OAI-compatible embedding creation options
   * @returns OAI-compatible embedding response
   */
  createEmbedding(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      if (!this.useEmbeddings) {
        throw new WllamaError(
          "Embeddings is not enabled. Please set it via LoadModelParams.embeddings"
        );
      }
      const result = yield this.proxy.wllamaAction(
        "embedding",
        {
          _name: "embd_req",
          data_json: JSON.stringify(options),
          files: []
          // TODO: support file input
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(options, false, result.req_id);
    });
  }
  /**
   * Rerank a list of documents against a query.
   * Requires the model to be loaded with embeddings: true and pooling_type: 'rank'.
   * @param options Reranking options (query, documents, top_n)
   * @returns Reranking response with relevance scores sorted highest first
   */
  createRerank(options) {
    return __async(this, null, function* () {
      var _a, _b;
      this.checkModelLoaded();
      if (!this.useEmbeddings || !this.useRerank) {
        throw new WllamaError(
          "Rerank is not enabled. Please set it via LoadModelParams: embeddings = true and pooling_type = rank"
        );
      }
      const top_n = (_a = options.top_n) != null ? _a : options.documents.length;
      let totalTokens = 0;
      const rawResults = [];
      for (let i = 0; i < options.documents.length; i++) {
        const result = yield this.proxy.wllamaAction("rerank", {
          _name: "rrnk_req",
          data_json: JSON.stringify({
            query: options.query,
            document: options.documents[i]
          })
        });
        if (!result.success) {
          throw new WllamaError(
            "Model failed to start reranking",
            "inference_error"
          );
        }
        const { score, tokens_evaluated } = yield this.getSingleResult(result.req_id, "reranking");
        totalTokens += tokens_evaluated;
        rawResults.push({ index: i, score });
      }
      rawResults.sort((a, b) => b.score - a.score);
      return {
        model: (_b = this.getModelMetadata().meta["general.name"]) != null ? _b : "",
        object: "list",
        usage: { prompt_tokens: totalTokens, total_tokens: totalTokens },
        results: rawResults.slice(0, top_n).map(({ index, score }) => ({
          index,
          relevance_score: score
        }))
      };
    });
  }
  /**
   * Answer typed questions about a state, using a decision model (TypeSafe-compatible System One API).
   * Image input is not supported yet.
   * @param options The state and the questions
   * @returns The answer of each question
   */
  createSystemOne(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "systemone",
        {
          _name: "sys1_req",
          data_json: JSON.stringify(options)
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start systemone",
          "inference_error"
        );
      }
      return yield this.getSingleResult(
        result.req_id,
        "systemone"
      );
    });
  }
  /**
   * Score candidate replies to chat messages (natlang decision readout): the log-probability of each reply as the closed final assistant message, summed over its tokens after the chat prompt. The replies of an item share the prompt.
   * @param options The items to score
   * @returns Per item, the log-probability and token count of each reply, or the error of an item that could not be rendered
   */
  createNatlangScore(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("score", {
        _name: "scor_req",
        data_json: JSON.stringify(options)
      });
      if (!result.success) {
        throw new WllamaError("Model failed to start scoring", "inference_error");
      }
      return yield this.getSingleResult(
        result.req_id,
        "score"
      );
    });
  }
  /**
   * Render chat messages (and tools) with the chat template, as the server's POST /apply-template does.
   * @param options OAI-compatible chat completion options (messages, tools, add_generation_prompt, ...)
   * @returns The prompt text
   */
  applyTemplate(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "apply_template",
        { _name: "tmpl_req", data_json: JSON.stringify(options) }
      );
      if (!result.success) {
        throw new WllamaError("Failed to apply chat template", "inference_error");
      }
      return result.prompt;
    });
  }
  /**
   * Save the state of a slot (its KV cache, tokens and context checkpoints), as the server's POST /slots/:id?action=save does. The slot must be idle.
   * @param idSlot The slot
   * @returns The saved state, to give to `restoreSlot` of an engine with the same model, build and context settings
   */
  saveSlot(idSlot) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const res = yield this.slotAction(
        idSlot,
        "save",
        filename
      );
      const state = yield this.proxy.slotFileRead(filename);
      return { state, n_saved: res.n_saved, n_written: res.n_written };
    });
  }
  /**
   * Restore a slot from a state of `saveSlot`, as the server's POST /slots/:id?action=restore does. A state that does not fit leaves the slot empty and throws.
   * @param idSlot The slot
   * @param state The saved state
   * @returns The number of restored tokens and bytes read
   */
  restoreSlot(idSlot, state) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        return yield this.slotAction(idSlot, "restore", filename);
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Clear the cached prompt of a slot, as the server's POST /slots/:id?action=erase does.
   * @param idSlot The slot
   */
  eraseSlot(idSlot) {
    return __async(this, null, function* () {
      return yield this.slotAction(idSlot, "erase", "");
    });
  }
  slotAction(idSlot, action, filename) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("slots", {
        _name: "slot_req",
        id_slot: idSlot,
        action,
        filename
      });
      if (!result.success) {
        throw new WllamaError(`Failed to ${action} slot`, "inference_error");
      }
      return yield this.getSingleResult(result.req_id, `slot ${action}`);
    });
  }
  /**
   * Store a natlang native state (natlang's store-file format, natlang.native-state/1) after the model is loaded, as the
   * server's PUT /v1/neuralese/native-states/:id does: the bytes are verified (layout, finite values, content ID), kept as
   * received and, for a state of this model's identity, registered so prompts consume it exactly.
   * @param state The state's bytes
   * @param id The ID the state must have (its content ID); omit to accept any
   * @returns The consumption mode and the state's metadata ({id, ...metadata})
   */
  putNativeState(state, id = "") {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const filename = `wllama-native-state-${++this.slotFileId}.nzs`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        const result = yield this.proxy.wllamaAction(
          "native_state",
          { _name: "nsta_req", action: "put", id, filename }
        );
        if (!result.success) {
          throw new WllamaError("Failed to put native state", "inference_error");
        }
        return yield this.getSingleResult(result.req_id, "native state put");
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Read a stored native state (one this engine produced, or one put or loaded), as the server's
   * GET /v1/neuralese/native-states/:id does: its bytes exactly as stored, natlang's store-file format.
   * @param id The state's ID
   * @returns The state's bytes and its metadata ({id, ...metadata})
   */
  getNativeState(id) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const filename = `wllama-native-state-${++this.slotFileId}.nzs`;
      const result = yield this.proxy.wllamaAction(
        "native_state",
        { _name: "nsta_req", action: "get", id, filename }
      );
      if (!result.success) {
        throw new WllamaError("Failed to get native state", "inference_error");
      }
      const state = yield this.proxy.slotFileRead(filename);
      return { state, meta: JSON.parse(result.meta_json) };
    });
  }
  createChatCompletion(options) {
    return __async(this, null, function* () {
      var _a;
      if (Object.keys(this.chatTemplateKwargs).length > 0) {
        options = __spreadProps(__spreadValues({}, options), {
          chat_template_kwargs: __spreadValues(__spreadValues({}, this.chatTemplateKwargs), (_a = options.chat_template_kwargs) != null ? _a : {})
        });
      }
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  createCompletion(options) {
    return __async(this, null, function* () {
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  /**
   * Private implementation of createCompletion
   */
  createCompletionImpl(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const isStream = !!options.stream;
      const isChat = !!options.messages;
      const customOpt = {};
      if (this.seed !== void 0) {
        customOpt.seed = this.seed;
      }
      let files = [];
      if (isChat) {
        const tmp = this.prepareMultimodalInput(
          options
        );
        options = tmp.params;
        files = tmp.files;
      }
      const result = yield this.proxy.wllamaAction(
        "completion",
        {
          _name: "cmpl_req",
          is_chat: isChat,
          data_json: JSON.stringify(__spreadValues(__spreadValues({}, options), customOpt)),
          files: files.map((f) => new Uint8Array(f))
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(
        options,
        isStream,
        result.req_id
      );
    });
  }
  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   * Only called when stream=true and no onData is provided.
   */
  createCompletionGenerator(options) {
    return new Promise((resolve) => {
      const createGenerator = cbToAsyncIter(
        (callback) => {
          this.createCompletionImpl(__spreadProps(__spreadValues({}, options), {
            onData: (chunk) => callback(chunk)
          })).then(() => callback(void 0, true)).catch((err) => callback(void 0, false, err));
        }
      );
      resolve(createGenerator());
    });
  }
  /**
   * Whether the currently loaded model supports a specific input modality (e.g. image or audio).
   * @param modality
   * @returns
   */
  supportInputModality(modality) {
    this.checkModelLoaded();
    if (modality === "image") {
      return !!this.loadedContextInfo.has_image_input;
    } else if (modality === "audio") {
      return !!this.loadedContextInfo.has_audio_input;
    } else {
      throw new WllamaError(
        "Unsupported modality: " + modality,
        "unknown_error"
      );
    }
  }
  /**
   * Unload the model and free all memory.
   *
   * Note: This function will NOT crash if model is not yet loaded
   */
  exit() {
    return __async(this, null, function* () {
      var _a;
      yield (_a = this.proxy) == null ? void 0 : _a.wllamaExit();
      this.proxy = null;
    });
  }
  /**
   * [FOR DEBUGGING ONLY] Run ggml backend ops tests without loading any model.
   *
   * Initializes the wasm runtime, executes `test-backend-ops` with the given args, then shuts down.
   *
   * For more info, please refer to guides/debug.md
   *
   * @param args Arguments forwarded to test-backend-ops (e.g. ["-o", "ADD"])
   * @returns retcode (0 = all tests passed) and success flag
   */
  testBackendOps() {
    return __async(this, arguments, function* (args = []) {
      var _a;
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (!(yield isSupportMultiThread())) {
        throw new WllamaError(
          "Multi-threading is required to run backend ops tests, but it is not supported in the current environment."
        );
      }
      const tmpProxy = new ProxyToWorker(
        this.getWorkerResources(),
        0,
        // single-thread; no model needed
        (_a = this.config.suppressNativeLog) != null ? _a : false,
        this.logger()
      );
      try {
        yield tmpProxy.moduleInit([]);
        const startResult = yield tmpProxy.wllamaStart();
        if (!startResult.success) {
          throw new WllamaError(
            `Error while calling start function, result = ${startResult}`
          );
        }
        const result = yield tmpProxy.wllamaAction(
          "test_backend_ops",
          { _name: "tbop_req", args: ["test-backend-ops", ...args] }
        );
        return { retcode: result.retcode, success: result.success };
      } finally {
        yield tmpProxy.wllamaExit();
      }
    });
  }
  //////////////////////////////////////////////
  // Low level API
  // TODO: add back
  /**
   * get debug info
   */
  _getDebugInfo() {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      return yield this.proxy.wllamaDebug();
    });
  }
  //////////////////////////////////////////////
  // Utils
  jsonDecode(data_json) {
    try {
      return JSON.parse(data_json);
    } catch (e) {
      this.logger().error("Failed to parse JSON:", data_json);
      throw new WllamaError("Failed to parse model output", "inference_error");
    }
  }
  prepareMultimodalInput(params) {
    const msg = params.messages;
    const msgNew = [];
    const files = [];
    for (const m of msg) {
      if (Array.isArray(m.content)) {
        const newContent = [];
        for (const c of m.content) {
          if (c.type === "text" || c.type === "native_state") {
            newContent.push(c);
          } else {
            if (!this.mediaMarker) {
              throw new WllamaError(
                "Media marker is undefined",
                "inference_error"
              );
            }
            files.push(c.data);
            newContent.push({
              type: "text",
              text: this.mediaMarker
            });
          }
        }
        msgNew.push(__spreadProps(__spreadValues({}, m), {
          content: newContent
        }));
      } else {
        msgNew.push(m);
      }
    }
    return {
      params: __spreadProps(__spreadValues({}, params), {
        messages: msgNew
      }),
      files
    };
  }
  // release the slot occupied by the request; cancelling an already-finished request is a no-op
  cancelRequest(reqId) {
    return __async(this, null, function* () {
      try {
        yield this.proxy.wllamaAction("cancel", {
          _name: "cncl_req",
          req_id: reqId
        });
      } catch (e) {
        this.logger().warn("Failed to cancel request", reqId, e);
      }
    });
  }
  getSingleResult(reqId, task) {
    return __async(this, null, function* () {
      let completed = false;
      try {
        while (true) {
          const chunk = yield this.proxy.wllamaAction(
            "get_result",
            { _name: "gres_req", req_id: reqId }
          );
          const jsonString = chunk.data_json;
          if (jsonString && jsonString.length > 0) {
            if (chunk.is_error) {
              const jsonData = this.jsonDecode(jsonString);
              throw new WllamaError(
                jsonData.message || `Unknown ${task} error`,
                "inference_error"
              );
            }
            completed = true;
            return this.jsonDecode(jsonString);
          }
          if (!chunk.has_more) {
            completed = true;
            break;
          }
        }
        throw new WllamaError(`No ${task} result received`, "inference_error");
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
    });
  }
  getResponse(options, isStream, reqId) {
    return __async(this, null, function* () {
      var _a, _b;
      let finalResult = null;
      let completed = false;
      try {
        while (true) {
          if ((_a = options.abortSignal) == null ? void 0 : _a.aborted) {
            throw new WllamaAbortError();
          }
          const result_chunk = yield this.proxy.wllamaAction(
            "get_result",
            {
              _name: "gres_req",
              req_id: reqId
            }
          );
          const jsonString = result_chunk.data_json;
          if (!jsonString || jsonString.length === 0) {
            if (!result_chunk.has_more) {
              completed = true;
              break;
            } else {
              continue;
            }
          }
          if (jsonString == "null") {
            continue;
          }
          let jsonData = this.jsonDecode(jsonString);
          finalResult = jsonData;
          if (result_chunk.is_error) {
            this.logger().error("Model returned an error:", jsonData);
            throw new WllamaError(
              jsonData.message || "Unknown inference error",
              "inference_error"
            );
          }
          if (isStream) {
            if (!Array.isArray(jsonData)) {
              jsonData = [jsonData];
            }
            for (const chunk of jsonData) {
              (_b = options.onData) == null ? void 0 : _b.call(options, chunk);
              finalResult = chunk;
            }
          }
          if (!result_chunk.has_more) {
            completed = true;
            break;
          }
        }
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
      return finalResult;
    });
  }
  getWorkerResources() {
    const workerResources = {
      wasmPath: absoluteUrl(this.pathConfig["default"]),
      compat: false
    };
    if (needCompat()) {
      if (!this.compat) {
        this.logger().warn(
          "Not using compat mode" + (isFirefox() ? " (expected on Firefox - WebGPU will be disabled)" : "")
        );
      } else {
        const isUsingDefault = this.compat.worker === WasmCompatFromCDN.worker && this.compat.wasm === WasmCompatFromCDN.wasm;
        if (isUsingDefault) {
          this.logger().warn(
            "Compatibility mode is activated, using resources from CDN. To use local resources, please refer to @wllama/wllama-compat package."
          );
          this.logger().warn(
            "IMPORTANT: Performance will be significantly degraded in compatibility mode."
          );
        }
        workerResources.wasmPath = absoluteUrl(this.compat.wasm);
        workerResources.jsPath = this.compat.worker;
        workerResources.compat = true;
      }
    }
    if (isFirefox()) {
      if (workerResources.compat) {
        this.logger().warn(
          'Using compat mode on Firefox, performance will be significantly degraded; Consider enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      } else if (!isSupportJSPI()) {
        this.logger().warn(
          'WebGPU is disabled on Firefox due to missing JSPI support. Please consider enabling compat mode, or enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      }
    }
    return workerResources;
  }
};
export {
  CacheManager,
  LogLevel,
  LoggerWithoutDebug,
  Model,
  ModelManager,
  ModelValidationStatus,
  POLYFILL_ETAG,
  Wllama,
  WllamaAbortError,
  WllamaError,
  WllamaRuntimeError,
  getHFFileSHA256,
  getHFModelSource,
  isValidGgufFile
};
